"""Unified training and calibration CLI for CXR Neuro-Symbolic Triage.

Supports local CPU smoke execution (with --max-batches and --no-pretrained)
and full Kaggle/cloud GPU acceleration without code modification.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from cxr_triage.calibration.calibrator import StudyCalibrator
from cxr_triage.config import NIH_CHESTXRAY14_LABELS, load_config
from cxr_triage.data.dataset import NIHChestXrayDataset, build_transforms
from cxr_triage.data.splits import compute_file_sha256
from cxr_triage.models.loss import TriageCompositeLoss
from cxr_triage.models.triage_net import TriageNet


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Unified CPU/GPU training and calibration CLI for CXR Triage."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to training configuration YAML.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to nih_splits.csv manifest (overrides config if provided).",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Path to raw image directory (overrides config if provided).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Neural backbone architecture name (overrides config if provided).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per training step.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional ceiling on batches per epoch for fast smoke verification.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained backbone weight downloads for offline/smoke runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/models",
        help="Directory to save model checkpoints and calibration parameters.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Hardware execution device target.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes (default 0 for local CPU safety; e.g. 4 on Kaggle GPU).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA FP16 Automatic Mixed Precision (autocast + GradScaler) when running on CUDA.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute training, post-hoc calibration, and artifact generation."""
    t0 = time.time()
    args = parse_args()

    # 1. Load frozen configuration
    config_path = Path(args.config)
    cfg = load_config(config_path)

    # 2. Select execution device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    print(
        f"[CXR Triage] Target Device: {device} | AMP: {use_amp} | Workers: {args.num_workers}",
        flush=True,
    )

    # 3. Resolve backbone
    backbone_name = args.backbone if args.backbone is not None else cfg.model.backbone
    print(
        f"[CXR Triage] Backbone: {backbone_name} (pretrained={not args.no_pretrained})",
        flush=True,
    )

    # 4. Construct dataset and transforms
    if args.manifest is not None:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = cfg.paths.processed_data_dir / "nih_splits.csv"
        if not manifest_path.exists():
            cxr_manifest = Path("C:/CXR/data/processed/nih_splits.csv")
            if cxr_manifest.exists():
                manifest_path = cxr_manifest

    if args.images_dir is not None:
        images_dir = Path(args.images_dir)
    else:
        images_dir = cfg.paths.nih_dir / "images"
        if not images_dir.exists():
            cxr_images = Path("C:/CXR/data/raw/nih/images")
            if cxr_images.exists():
                images_dir = cxr_images

    ontology_labels = cfg.ontology.labels

    train_transform = build_transforms(cfg.model.image_size, is_train=True)
    eval_transform = build_transforms(cfg.model.image_size, is_train=False)

    train_dataset = NIHChestXrayDataset(
        manifest_path=manifest_path,
        images_dir=images_dir,
        split="train",
        ontology_labels=ontology_labels,
        transform=train_transform,
        only_existing=True,
    )

    calib_dataset = NIHChestXrayDataset(
        manifest_path=manifest_path,
        images_dir=images_dir,
        split="calibration",
        ontology_labels=ontology_labels,
        transform=eval_transform,
        only_existing=True,
    )

    print(
        f"[CXR Triage] Dataset Loaded: Train={len(train_dataset)} samples, "
        f"Calibration={len(calib_dataset)} samples",
        flush=True,
    )

    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # 5. Instantiate model, loss, and optimizer
    model = TriageNet(
        backbone_name=backbone_name,
        num_classes=len(NIH_CHESTXRAY14_LABELS),
        pretrained=not args.no_pretrained,
        proj_dim=512,
        severity_classes=3,
    ).to(device)

    param_counts = model.count_parameters()
    print(
        f"[CXR Triage] Model Parameters: Total={param_counts['total']:,}, "
        f"Inference={param_counts['inference_only']:,}",
        flush=True,
    )

    criterion = TriageCompositeLoss(
        lambda_s=cfg.distillation.lambda_s,
        lambda_d=0.0,  # Distillation loss active when multimodal teacher is provided
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"

    # 6. Training loop
    model.train()
    final_loss = 0.0
    total_batches = (
        min(len(train_loader), args.max_batches)
        if args.max_batches is not None
        else len(train_loader)
    )
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        batches_processed = 0
        for batch_idx, batch in enumerate(train_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            images = batch["image"].to(device, non_blocking=pin_memory)
            labels = batch["labels"].to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            amp_ctx: contextlib.AbstractContextManager[Any] = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else contextlib.nullcontext()
            )
            with amp_ctx:
                outputs = model(images)
                loss_dict = criterion(
                    cls_logits=outputs["cls_logits"],
                    labels=labels,
                    struct_logits=outputs["struct_logits"],
                )
                loss = loss_dict["loss"]

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item())
            batches_processed += 1

            if batches_processed % 250 == 0:
                avg_loss = running_loss / batches_processed
                elapsed = time.time() - epoch_start
                print(
                    f"[CXR Triage] Epoch {epoch}/{args.epochs} | "
                    f"Batch {batches_processed}/{total_batches} | "
                    f"Loss: {avg_loss:.4f} | Elapsed: {elapsed:.1f}s",
                    flush=True,
                )

        final_loss = running_loss / max(1, batches_processed)
        epoch_elapsed = time.time() - epoch_start
        print(
            f"[CXR Triage] Epoch {epoch}/{args.epochs} - Loss: {final_loss:.4f} "
            f"({batches_processed} batches, {epoch_elapsed:.1f}s)",
            flush=True,
        )

        # Persist checkpoint at the end of every epoch to guard against session interruption
        torch.save(
            {
                "epoch": epoch,
                "backbone": backbone_name,
                "model_state_dict": model.state_dict(),
                "param_counts": param_counts,
            },
            checkpoint_path,
        )

    # 7. Calibration phase
    print("[CXR Triage] Fitting post-hoc StudyCalibrator on calibration split...", flush=True)
    model.eval()
    calib_logits_list: list[np.ndarray] = []
    calib_labels_list: list[np.ndarray] = []
    calib_y_study_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(calib_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=pin_memory)
            outputs = model(images, inference_only=True)
            calib_logits_list.append(outputs["cls_logits"].cpu().numpy())
            calib_labels_list.append(batch["labels"].cpu().numpy())
            calib_y_study_list.append(batch["y_study"].cpu().numpy())

    if calib_logits_list:
        calib_logits = np.concatenate(calib_logits_list, axis=0)
        calib_labels = np.concatenate(calib_labels_list, axis=0)
        calib_y_study = np.concatenate(calib_y_study_list, axis=0)
        calibrator = StudyCalibrator().fit(calib_logits, calib_labels, calib_y_study)
    else:
        # Fallback identity calibrator if calibration set is empty in mock/smoke setup
        calibrator = StudyCalibrator(temperature=1.0, study_scale=1.0, study_bias=0.0)

    # 8. Save artifacts
    # Save final checkpoint
    torch.save(
        {
            "epoch": args.epochs,
            "backbone": backbone_name,
            "model_state_dict": model.state_dict(),
            "param_counts": param_counts,
        },
        checkpoint_path,
    )
    checkpoint_hash = compute_file_sha256(checkpoint_path)

    # Save calibrator JSON
    calibrator_path = output_dir / "calibrator.json"
    calibrator_hash = calibrator.save(calibrator_path)

    # Save run summary
    total_duration = time.time() - t0
    run_summary: dict[str, Any] = {
        "status": "COMPLETED",
        "device": str(device),
        "backbone": backbone_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_batches": args.max_batches,
        "num_workers": args.num_workers,
        "amp": use_amp,
        "final_train_loss": float(final_loss),
        "parameter_counts": param_counts,
        "calibration": {
            "temperature": float(calibrator.temperature),
            "study_scale": float(calibrator.study_scale),
            "study_bias": float(calibrator.study_bias),
            "sha256": calibrator_hash,
        },
        "artifacts": {
            "checkpoint_pt": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "calibrator_json": str(calibrator_path.resolve()),
            "calibrator_sha256": calibrator_hash,
        },
        "duration_seconds": float(total_duration),
    }

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(f"[CXR Triage] Saved artifacts to {output_dir} in {total_duration:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
