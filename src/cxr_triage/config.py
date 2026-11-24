"""Typed, immutable configuration loader with strict runtime validation for CXR Triage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

NIH_CHESTXRAY14_LABELS: frozenset[str] = frozenset({
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
})


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    version: str
    description: str


@dataclass(frozen=True)
class PathsConfig:
    raw_data_dir: Path
    nih_dir: Path
    processed_data_dir: Path
    artifacts_dir: Path
    models_dir: Path
    eval_dir: Path
    root_dir: Path


@dataclass(frozen=True)
class ReproducibilityConfig:
    seed: int
    deterministic: bool


@dataclass(frozen=True)
class SplitsConfig:
    train: float
    val: float
    test: float


@dataclass(frozen=True)
class DataConfig:
    splits: SplitsConfig


@dataclass(frozen=True)
class OntologyConfig:
    dataset_name: str
    labels: tuple[str, ...]
    critical_findings: tuple[str, ...]


@dataclass(frozen=True)
class StudyParametersConfig:
    epsilon: float
    delta: float
    c_min: float
    delta_c: float
    delta_tol: float
    sigma_max: float
    coverage_grid: tuple[float, ...]


@dataclass(frozen=True)
class ModelConfig:
    backbone: str
    pretrained: bool
    image_size: int
    in_channels: int
    num_classes: int


@dataclass(frozen=True)
class DistillationConfig:
    teacher: str
    lambda_s: float
    lambda_d: float


@dataclass(frozen=True)
class CalibrationConfig:
    method: str
    eval_ece_bins: int


@dataclass(frozen=True)
class RulesConfig:
    enable_symbolic_audit: bool
    critical_escalation_threshold: float
    defer_on_high_uncertainty: bool


@dataclass(frozen=True)
class ApiConfig:
    host: str
    port: int
    reload: bool


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    paths: PathsConfig
    reproducibility: ReproducibilityConfig
    data: DataConfig
    ontology: OntologyConfig
    study_parameters: StudyParametersConfig
    model: ModelConfig
    distillation: DistillationConfig
    calibration: CalibrationConfig
    rules: RulesConfig
    api: ApiConfig


def _resolve_path(root: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (root / p).resolve()


def validate_config(config: AppConfig) -> None:
    """Validate configuration semantics and invariant bounds."""
    # 1. Splits validation
    splits = config.data.splits
    if splits.train <= 0.0 or splits.val <= 0.0 or splits.test <= 0.0:
        raise ValueError(
            f"All split ratios must be positive (> 0.0), got train={splits.train}, "
            f"val={splits.val}, test={splits.test}"
        )
    total_splits = splits.train + splits.val + splits.test
    if not math.isclose(total_splits, 1.0, abs_tol=1e-5):
        raise ValueError(f"Split ratios must sum to 1.0, got sum={total_splits:.6f}")

    # 2. Study parameters validation
    params = config.study_parameters
    if not (0.0 < params.epsilon < 1.0):
        raise ValueError(f"epsilon must be strictly between 0 and 1, got {params.epsilon}")
    if not (0.0 < params.delta < 1.0):
        raise ValueError(f"delta must be strictly between 0 and 1, got {params.delta}")
    if not (0.0 < params.c_min <= 1.0):
        raise ValueError(f"c_min must be in (0.0, 1.0], got {params.c_min}")
    if params.delta_c <= 0.0:
        raise ValueError(f"delta_c must be positive, got {params.delta_c}")
    if params.delta_tol < 0.0:
        raise ValueError(f"delta_tol must be non-negative, got {params.delta_tol}")
    if params.sigma_max <= 0.0:
        raise ValueError(f"sigma_max must be positive, got {params.sigma_max}")
    if not params.coverage_grid:
        raise ValueError("coverage_grid must not be empty")
    for cov in params.coverage_grid:
        if not (0.0 < cov <= 1.0):
            raise ValueError(f"Coverage grid values must be in (0.0, 1.0], got {cov}")

    # 3. Ontology validation
    ont = config.ontology
    if len(ont.labels) != 14:
        raise ValueError(f"Expected 14 labels for NIH-ChestXray14, got {len(ont.labels)}")
    if set(ont.labels) != NIH_CHESTXRAY14_LABELS:
        missing = NIH_CHESTXRAY14_LABELS - set(ont.labels)
        extra = set(ont.labels) - NIH_CHESTXRAY14_LABELS
        raise ValueError(
            f"Labels do not match standard NIH-ChestXray14 ontology. Missing: {missing}, Extra: {extra}"
        )
    if not ont.critical_findings:
        raise ValueError("critical_findings cannot be empty")
    crit_set = set(ont.critical_findings)
    if not crit_set.issubset(NIH_CHESTXRAY14_LABELS):
        invalid_crit = crit_set - NIH_CHESTXRAY14_LABELS
        raise ValueError(
            f"critical_findings must be a subset of ontology labels, found invalid: {invalid_crit}"
        )

    # 4. Model validation
    model = config.model
    if model.in_channels not in (1, 3):
        raise ValueError(f"in_channels must be 1 or 3, got {model.in_channels}")
    if model.image_size <= 0:
        raise ValueError(f"image_size must be positive, got {model.image_size}")
    if model.num_classes != len(ont.labels):
        raise ValueError(
            f"num_classes ({model.num_classes}) must match label count ({len(ont.labels)})"
        )

    # 5. Distillation validation
    dist = config.distillation
    if not dist.teacher.strip():
        raise ValueError("Distillation teacher must be a non-empty string")
    if dist.lambda_s < 0.0:
        raise ValueError(f"lambda_s must be non-negative, got {dist.lambda_s}")
    if dist.lambda_d < 0.0:
        raise ValueError(f"lambda_d must be non-negative, got {dist.lambda_d}")

    # 6. API validation
    api = config.api
    if not (1 <= api.port <= 65535):
        raise ValueError(f"api.port must be between 1 and 65535, got {api.port}")
    if not api.host.strip():
        raise ValueError("api.host must be a non-empty string")


def load_config(
    config_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> AppConfig:
    """Load, dynamically resolve paths, and validate the application configuration.

    Args:
        config_path: Path to the YAML configuration file. If None, resolves to
            `configs/default.yaml` relative to `project_root`.
        project_root: Root directory of the project. If None, dynamically resolved
            from `config_path` or module location.

    Returns:
        An immutable AppConfig instance.
    """
    if project_root is not None:
        root = Path(project_root).resolve()
    elif config_path is not None:
        root = Path(config_path).resolve().parent.parent
    else:
        # Default: current module is in src/cxr_triage/config.py -> parents[2] is root
        root = Path(__file__).resolve().parents[2]

    if config_path is None:
        cfg_file = (root / "configs" / "default.yaml").resolve()
    else:
        cfg_file = Path(config_path).resolve()

    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found at: {cfg_file}")

    with open(cfg_file, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    paths_data = data.get("paths", {})
    paths_cfg = PathsConfig(
        raw_data_dir=_resolve_path(root, paths_data.get("raw_data_dir", "data/raw")),
        nih_dir=_resolve_path(root, paths_data.get("nih_dir", "data/raw/nih")),
        processed_data_dir=_resolve_path(
            root, paths_data.get("processed_data_dir", "data/processed")
        ),
        artifacts_dir=_resolve_path(root, paths_data.get("artifacts_dir", "artifacts")),
        models_dir=_resolve_path(root, paths_data.get("models_dir", "artifacts/models")),
        eval_dir=_resolve_path(root, paths_data.get("eval_dir", "artifacts/evaluation")),
        root_dir=root,
    )

    proj_data = data.get("project", {})
    proj_cfg = ProjectConfig(
        name=str(proj_data.get("name", "")),
        version=str(proj_data.get("version", "")),
        description=str(proj_data.get("description", "")),
    )

    repro_data = data.get("reproducibility", {})
    repro_cfg = ReproducibilityConfig(
        seed=int(repro_data.get("seed", 42)),
        deterministic=bool(repro_data.get("deterministic", True)),
    )

    data_section = data.get("data", {})
    splits_data = data_section.get("splits", {})
    splits_cfg = SplitsConfig(
        train=float(splits_data.get("train", 0.70)),
        val=float(splits_data.get("val", 0.15)),
        test=float(splits_data.get("test", 0.15)),
    )
    data_cfg = DataConfig(splits=splits_cfg)

    ont_data = data.get("ontology", {})
    ont_cfg = OntologyConfig(
        dataset_name=str(ont_data.get("dataset_name", "")),
        labels=tuple(ont_data.get("labels", ())),
        critical_findings=tuple(ont_data.get("critical_findings", ())),
    )

    study_data = data.get("study_parameters", {})
    study_cfg = StudyParametersConfig(
        epsilon=float(study_data.get("epsilon", 0.02)),
        delta=float(study_data.get("delta", 0.05)),
        c_min=float(study_data.get("c_min", 0.30)),
        delta_c=float(study_data.get("delta_c", 0.10)),
        delta_tol=float(study_data.get("delta_tol", 0.05)),
        sigma_max=float(study_data.get("sigma_max", 0.10)),
        coverage_grid=tuple(float(c) for c in study_data.get("coverage_grid", ())),
    )

    model_data = data.get("model", {})
    model_cfg = ModelConfig(
        backbone=str(model_data.get("backbone", "")),
        pretrained=bool(model_data.get("pretrained", True)),
        image_size=int(model_data.get("image_size", 224)),
        in_channels=int(model_data.get("in_channels", 3)),
        num_classes=int(model_data.get("num_classes", 14)),
    )

    dist_data = data.get("distillation", {})
    dist_cfg = DistillationConfig(
        teacher=str(dist_data.get("teacher", "BiomedCLIP")),
        lambda_s=float(dist_data.get("lambda_s", 0.5)),
        lambda_d=float(dist_data.get("lambda_d", 0.3)),
    )

    calib_data = data.get("calibration", {})
    calib_cfg = CalibrationConfig(
        method=str(calib_data.get("method", "temperature_scaling")),
        eval_ece_bins=int(calib_data.get("eval_ece_bins", 15)),
    )

    rules_data = data.get("rules", {})
    rules_cfg = RulesConfig(
        enable_symbolic_audit=bool(rules_data.get("enable_symbolic_audit", True)),
        critical_escalation_threshold=float(
            rules_data.get("critical_escalation_threshold", 0.35)
        ),
        defer_on_high_uncertainty=bool(
            rules_data.get("defer_on_high_uncertainty", True)
        ),
    )

    api_data = data.get("api", {})
    api_cfg = ApiConfig(
        host=str(api_data.get("host", "127.0.0.1")),
        port=int(api_data.get("port", 8000)),
        reload=bool(api_data.get("reload", False)),
    )

    app_config = AppConfig(
        project=proj_cfg,
        paths=paths_cfg,
        reproducibility=repro_cfg,
        data=data_cfg,
        ontology=ont_cfg,
        study_parameters=study_cfg,
        model=model_cfg,
        distillation=dist_cfg,
        calibration=calib_cfg,
        rules=rules_cfg,
        api=api_cfg,
    )

    validate_config(app_config)
    return app_config
