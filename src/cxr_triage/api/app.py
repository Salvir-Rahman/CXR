"""FastAPI inference backend serving the Neuro-Symbolic Chest X-Ray Triage pipeline.

Exposes:
    - GET /health: Service health status, model metadata, and rule registry provenance.
    - POST /predict: Single-study image triage with calibrated finding probabilities,
      Horn clause clinical rule auditing, and neuro-symbolic escalation vetoes.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from cxr_triage.calibration.calibrator import StudyCalibrator
from cxr_triage.config import NIH_CHESTXRAY14_LABELS, load_config
from cxr_triage.data.dataset import build_transforms
from cxr_triage.models.triage_net import TriageNet
from cxr_triage.rules.engine import evaluate_rules, load_rule_registry

DISCLAIMER_TEXT: str = (
    "Research prototype only. Not a medical device and not for clinical decision-making."
)


class HealthResponse(BaseModel):
    """Health check and model provenance response schema."""

    status: str = Field(default="healthy", description="Operational status.")
    model_backbone: str = Field(..., description="Neural backbone architecture name.")
    rules_loaded: int = Field(..., description="Number of loaded clinical Horn clauses.")
    rules_sha256: str = Field(..., description="SHA-256 fingerprint of the rule registry.")
    device: str = Field(..., description="Hardware execution device.")
    disclaimer: str = Field(default=DISCLAIMER_TEXT, description="Clinical safety disclaimer.")


class TriageResponse(BaseModel):
    """Complete study triage decision and interpretability payload."""

    decision: Literal["auto", "esc"] = Field(
        ..., description="Triage action: 'auto' (autonomous normal report) or 'esc' (escalate to radiologist)."
    )
    study_abnormality_score: float = Field(
        ..., description="Calibrated posterior study-level abnormality probability in [0, 1]."
    )
    threshold_tau: float = Field(
        ..., description="Risk threshold tau applied for autonomous triage."
    )
    finding_probabilities: dict[str, float] = Field(
        ..., description="Calibrated probabilities for each of the 14 NIH ontology findings."
    )
    discretized_findings: dict[str, str] = Field(
        ..., description="Discretized severity state ('absent', 'present', 'severe') per finding."
    )
    supporting_rules: list[str] = Field(
        ..., description="List of rule IDs whose antecedents and consequents were satisfied."
    )
    violated_rules: list[str] = Field(
        ..., description="List of rule IDs whose antecedents held but consequents were violated."
    )
    escalation_reasons: list[str] = Field(
        ..., description="Clinically auditable reasons explaining escalation to human review."
    )
    inference_latency_ms: float = Field(
        ..., description="Total wall-clock inference and symbolic evaluation latency in milliseconds."
    )
    disclaimer: str = Field(default=DISCLAIMER_TEXT, description="Clinical safety disclaimer.")


class TriageService:
    """Core stateful inference service managing model, calibrator, and symbolic rules."""

    def __init__(
        self,
        config_path: Path | None = None,
        rules_path: Path | None = None,
        checkpoint_path: Path | None = None,
        calibrator_path: Path | None = None,
    ) -> None:
        """Initialize the triage service and load models and rule registries.

        Args:
            config_path: Optional path to default.yaml configuration.
            rules_path: Optional path to rules-v1.0.json registry.
            checkpoint_path: Optional path to trained model checkpoint.pt.
            calibrator_path: Optional path to calibrator.json parameters.
        """
        # Resolve configuration
        cfg_file = config_path or Path("configs/default.yaml")
        if not cfg_file.exists():
            cfg_file = Path("C:/CXR/configs/default.yaml")
        self.cfg = load_config(cfg_file)

        # Resolve rules registry
        rule_file = rules_path or Path("configs/rules-v1.0.json")
        if not rule_file.exists():
            rule_file = Path("C:/CXR/configs/rules-v1.0.json")
        self.rules, self.rules_sha256 = load_rule_registry(
            rule_file, frozenset(NIH_CHESTXRAY14_LABELS)
        )

        # Execution device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone_name = self.cfg.model.backbone
        self.ontology_labels = list(self.cfg.ontology.labels)

        # Transform pipeline
        self.transform = build_transforms(self.cfg.model.image_size, is_train=False)

        # Instantiate neural model
        self.model = TriageNet(
            backbone_name=self.backbone_name,
            num_classes=len(self.ontology_labels),
            pretrained=False,
            proj_dim=512,
            severity_classes=3,
        ).to(self.device)

        # Resolve checkpoint path from arg or env
        ckpt_path_str = os.getenv("CXR_CHECKPOINT_PATH")
        resolved_ckpt = checkpoint_path or (Path(ckpt_path_str) if ckpt_path_str else None)
        if resolved_ckpt and resolved_ckpt.exists():
            checkpoint = torch.load(resolved_ckpt, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)

        self.model.eval()

        # Resolve calibrator path from arg or env
        calib_path_str = os.getenv("CXR_CALIBRATOR_PATH")
        resolved_calib = calibrator_path or (Path(calib_path_str) if calib_path_str else None)
        if resolved_calib and resolved_calib.exists():
            self.calibrator = StudyCalibrator.load(resolved_calib)
        else:
            self.calibrator = StudyCalibrator(temperature=1.0, study_scale=1.0, study_bias=0.0)

        self.disclaimer = DISCLAIMER_TEXT

    def predict_image(self, pil_image: Image.Image, tau: float = 0.25) -> TriageResponse:
        """Run complete inference and neuro-symbolic audit pipeline on a single chest X-ray.

        Args:
            pil_image: PIL Image instance of the radiograph.
            tau: Abnormality risk score threshold.

        Returns:
            TriageResponse payload with decision, probabilities, and rule audits.
        """
        t_start = time.perf_counter()

        # Ensure image is in 3-channel RGB mode
        img_rgb = pil_image.convert("RGB")
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        # Neural forward pass
        with torch.no_grad():
            outputs = self.model(img_tensor, inference_only=True)
            cls_logits_t = outputs["cls_logits"]
            struct_logits_t = outputs["struct_logits"]

            if cls_logits_t is None or struct_logits_t is None:
                raise RuntimeError("Model forward pass failed to generate logits.")

            cls_logits = cls_logits_t.cpu().numpy()
            struct_logits = struct_logits_t.cpu().numpy()

        # Post-hoc calibration
        p_findings, p_study = self.calibrator.predict_calibrated(cls_logits)
        finding_probs = {
            label: float(p_findings[0, i]) for i, label in enumerate(self.ontology_labels)
        }
        study_score = float(p_study[0])

        # Discretize 3-state severity predictions (0: absent, 1: present, 2: severe)
        state_labels = {0: "absent", 1: "present", 2: "severe"}
        discrete_idx = np.argmax(struct_logits[0], axis=-1)  # shape (14,)
        discretized_findings = {
            label: state_labels[int(discrete_idx[i])]
            for i, label in enumerate(self.ontology_labels)
        }

        # Symbolic Horn clause rule evaluation
        rule_res = evaluate_rules(discretized_findings, self.rules)
        supporting_rules = list(rule_res.supporting_rule_ids)
        violated_rules = list(rule_res.violated_rule_ids)

        # Neuro-symbolic decision logic
        escalation_reasons: list[str] = []
        if study_score > tau:
            escalation_reasons.append(
                f"Calibrated study abnormality risk score ({study_score:.4f}) exceeds threshold tau ({tau:.4f})."
            )
        if rule_res.is_violated:
            escalation_reasons.append(
                f"Clinical guideline rule consistency violated: {', '.join(violated_rules)}."
            )

        decision: Literal["auto", "esc"] = "esc" if escalation_reasons else "auto"
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return TriageResponse(
            decision=decision,
            study_abnormality_score=study_score,
            threshold_tau=float(tau),
            finding_probabilities=finding_probs,
            discretized_findings=discretized_findings,
            supporting_rules=supporting_rules,
            violated_rules=violated_rules,
            escalation_reasons=escalation_reasons,
            inference_latency_ms=float(latency_ms),
            disclaimer=self.disclaimer,
        )


def create_app(
    config_path: Path | None = None,
    rules_path: Path | None = None,
    checkpoint_path: Path | None = None,
    calibrator_path: Path | None = None,
) -> FastAPI:
    """Application factory for the CXR Triage FastAPI backend.

    Args:
        config_path: Optional configuration YAML path.
        rules_path: Optional rules-v1.0.json path.
        checkpoint_path: Optional checkpoint.pt path.
        calibrator_path: Optional calibrator.json path.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Neuro-Symbolic Chest X-Ray Triage API",
        version="1.0.0",
        description=(
            "Production-grade, sub-millisecond inference API serving the Neuro-Symbolic "
            "Chest X-Ray Triage pipeline with clinical guideline consistency auditing."
        ),
    )

    # Configure CORS for local development and demo hosting
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize service
    service = TriageService(
        config_path=config_path,
        rules_path=rules_path,
        checkpoint_path=checkpoint_path,
        calibrator_path=calibrator_path,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """System health and rule registry provenance check."""
        return HealthResponse(
            status="healthy",
            model_backbone=service.backbone_name,
            rules_loaded=len(service.rules),
            rules_sha256=service.rules_sha256,
            device=str(service.device),
            disclaimer=service.disclaimer,
        )

    @app.post("/predict", response_model=TriageResponse)
    async def predict(
        file: Annotated[UploadFile, File(description="Chest X-ray image (PNG/JPEG/DICOM).")],
        tau: Annotated[
            float,
            Query(
                ge=0.0,
                le=1.0,
                description="Abnormality risk score threshold for autonomous triage.",
            ),
        ] = 0.25,
    ) -> TriageResponse:
        """Run single-study neuro-symbolic triage on an uploaded radiograph."""
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Empty image payload received. Please upload a valid image file.",
            )

        try:
            with Image.open(io.BytesIO(contents)) as pil_img:
                # Force PIL to decode the image buffer to catch corrupt streams
                pil_img.load()
                return service.predict_image(pil_img, tau=tau)
        except (UnidentifiedImageError, ValueError, OSError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded file is not a valid or readable image: {exc}",
            ) from exc

    return app


# Expose default application instance for uvicorn
app = create_app()
