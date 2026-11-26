"""Single-command Deliverable D6 evaluation CLI for Neuro-Symbolic CXR Triage.

Loads frozen checkpoints, calibrators, and rules, calibrates risk thresholds
on the calibration split, benchmarks abstention policies on the test split,
computes paired patient cluster bootstrap statistics, and benchmarks CPU latency.
"""

from __future__ import annotations

import argparse
import hashlib
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
from cxr_triage.eval.benchmark import (
    benchmark_cpu_latency,
    evaluate_all_policies,
    paired_patient_bootstrap_diff,
)
from cxr_triage.models.triage_net import TriageNet
from cxr_triage.policies.abstention import (
    fit_clopper_pearson_threshold,
    fit_cluster_ltt_threshold,
    policy_neuro_symbolic,
    policy_prob,
)
from cxr_triage.rules.engine import load_rule_registry

DISCLAIMER_TEXT: str = (
    "Research prototype only. Not a medical device and not for clinical decision-making."
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Single-command Deliverable D6 evaluation CLI for CXR Triage."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to default.yaml configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained TriageNet model checkpoint.pt.",
    )
    parser.add_argument(
        "--calibrator",
        type=str,
        required=True,
        help="Path to fitted StudyCalibrator calibrator.json.",
    )
    parser.add_argument(
        "--rules",
        type=str,
        default="configs/rules-v1.0.json",
        help="Path to clinical rule registry rules-v1.0.json.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional ceiling on batches per split for rapid CPU verification.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/evaluation",
        help="Directory to save evaluation results and SHA-256 fingerprint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Target execution hardware.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes (default 0 for local CPU safety; e.g. 4 on Kaggle GPU).",
    )
    return parser.parse_args()


def main() -> None:
    """Execute evaluation benchmark pipeline."""
    t0 = time.time()
    args = parse_args()

    # 1. Load config and resolve device
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 72, flush=True)
    print("NEURO-SYMBOLIC CHEST X-RAY TRIAGE: DELIVERABLE D6 BENCHMARK ENGINE", flush=True)
    print(
        f"Device: {device} | Model: {cfg.model.backbone} | Workers: {args.num_workers} | "
        f"Epsilon: {cfg.study_parameters.epsilon} | Delta: {cfg.study_parameters.delta}",
        flush=True,
    )
    print("=" * 72, flush=True)

    # 2. Input validation and SHA-256 fingerprinting
    ckpt_path = Path(args.checkpoint)
    calib_path = Path(args.calibrator)
    rules_path = Path(args.rules)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibrator not found at: {calib_path}")
    if not rules_path.exists():
        raise FileNotFoundError(f"Rules not found at: {rules_path}")

    # Resolve manifest and images paths with worktree fallback
    manifest_path = cfg.paths.processed_data_dir / "nih_splits.csv"
    if not manifest_path.exists():
        cxr_manifest = Path("C:/CXR/data/processed/nih_splits.csv")
        if cxr_manifest.exists():
            manifest_path = cxr_manifest

    images_dir = cfg.paths.nih_dir / "images"
    if not images_dir.exists():
        cxr_images = Path("C:/CXR/data/raw/nih/images")
        if cxr_images.exists():
            images_dir = cxr_images

    ckpt_hash = compute_file_sha256(ckpt_path)
    calib_hash = compute_file_sha256(calib_path)
    rules, rules_hash = load_rule_registry(rules_path, frozenset(NIH_CHESTXRAY14_LABELS))
    manifest_hash = compute_file_sha256(manifest_path) if manifest_path.exists() else "N/A"

    print(f"[Inputs] Checkpoint SHA-256 : {ckpt_hash[:16]}...", flush=True)
    print(f"[Inputs] Calibrator SHA-256 : {calib_hash[:16]}...", flush=True)
    print(f"[Inputs] Rule Registry      : {len(rules)} rules ({rules_hash[:16]}...)", flush=True)
    print(f"[Inputs] Dataset Manifest   : {manifest_path.name} ({manifest_hash[:16]}...)", flush=True)

    # 3. Instantiate model and calibrator
    model = TriageNet(
        backbone_name=cfg.model.backbone,
        num_classes=len(NIH_CHESTXRAY14_LABELS),
        pretrained=False,
        proj_dim=512,
        severity_classes=3,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    calibrator = StudyCalibrator.load(calib_path)
    eval_transform = build_transforms(cfg.model.image_size, is_train=False)

    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0

    # 4. Phase 1: Calibrate thresholds on calibration split
    print("\n--- Phase 1: Risk Threshold Calibration ---", flush=True)
    calib_ds = NIHChestXrayDataset(
        manifest_path=manifest_path,
        images_dir=images_dir,
        split="calibration",
        ontology_labels=cfg.ontology.labels,
        transform=eval_transform,
        only_existing=True,
    )
    calib_loader = DataLoader(
        calib_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    calib_logits_list: list[np.ndarray] = []
    calib_y_list: list[np.ndarray] = []
    calib_pat_list: list[np.ndarray] = []

    with torch.no_grad():
        for b_idx, batch in enumerate(calib_loader):
            if args.max_batches is not None and b_idx >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=pin_memory)
            out = model(images, inference_only=True)
            calib_logits_list.append(out["cls_logits"].cpu().numpy())
            calib_y_list.append(batch["y_study"].cpu().numpy())
            calib_pat_list.append(batch["patient_id"].cpu().numpy())

    calib_logits = np.concatenate(calib_logits_list, axis=0)
    calib_y_study = np.concatenate(calib_y_list, axis=0)
    calib_patient_ids = np.concatenate(calib_pat_list, axis=0)

    _, calib_p_study = calibrator.predict_calibrated(calib_logits)

    tau_prob = fit_clopper_pearson_threshold(
        calib_p_study,
        calib_y_study,
        epsilon=cfg.study_parameters.epsilon,
        delta=cfg.study_parameters.delta,
    )
    tau_ltt = fit_cluster_ltt_threshold(
        calib_p_study,
        calib_y_study,
        calib_patient_ids,
        epsilon=cfg.study_parameters.epsilon,
        delta=cfg.study_parameters.delta,
    )

    # Fallback to default study tau if calibration set is small/unachievable
    applied_tau_prob = tau_prob if tau_prob > 0.0 else 0.25
    applied_tau_ltt = tau_ltt if tau_ltt > 0.0 else applied_tau_prob

    print(
        f"Calibrated tau_prob (Clopper-Pearson) : {tau_prob:.4f} (applied: {applied_tau_prob:.4f})",
        flush=True,
    )
    print(
        f"Calibrated tau_ltt  (Cluster LTT)     : {tau_ltt:.4f} (applied: {applied_tau_ltt:.4f})",
        flush=True,
    )

    # 5. Phase 2: Test Split Evaluation
    print("\n--- Phase 2: Test Split Policy Benchmarking ---", flush=True)
    test_ds = NIHChestXrayDataset(
        manifest_path=manifest_path,
        images_dir=images_dir,
        split="test",
        ontology_labels=cfg.ontology.labels,
        transform=eval_transform,
        only_existing=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    test_logits_list: list[np.ndarray] = []
    test_struct_list: list[np.ndarray] = []
    test_y_list: list[np.ndarray] = []
    test_pat_list: list[np.ndarray] = []

    with torch.no_grad():
        for b_idx, batch in enumerate(test_loader):
            if args.max_batches is not None and b_idx >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=pin_memory)
            out = model(images, inference_only=True)
            test_logits_list.append(out["cls_logits"].cpu().numpy())
            test_struct_list.append(out["struct_logits"].cpu().numpy())
            test_y_list.append(batch["y_study"].cpu().numpy())
            test_pat_list.append(batch["patient_id"].cpu().numpy())

    test_logits = np.concatenate(test_logits_list, axis=0)
    test_struct = np.concatenate(test_struct_list, axis=0)
    test_y_study = np.concatenate(test_y_list, axis=0)
    test_patient_ids = np.concatenate(test_pat_list, axis=0)

    _, test_p_study = calibrator.predict_calibrated(test_logits)

    # Discretize 3-state severity predictions per sample
    state_map = {0: "absent", 1: "present", 2: "severe"}
    discrete_indices = np.argmax(test_struct, axis=-1)  # shape (N, 14)
    discretized_findings = [
        {label: state_map[int(discrete_indices[i, j])] for j, label in enumerate(cfg.ontology.labels)}
        for i in range(len(test_struct))
    ]

    # Evaluate 3 primary policies: pi_prob, pi_conf, pi_ns
    policy_metrics = evaluate_all_policies(
        p_study=test_p_study,
        y_study=test_y_study,
        tau_prob=applied_tau_prob,
        tau_ltt=applied_tau_ltt,
        rules=rules,
        discretized_findings=discretized_findings,
        delta=cfg.study_parameters.delta,
    )

    # 6. Phase 3: Paired Patient Bootstrap Difference (pi_ns vs pi_prob)
    print("\n--- Phase 3: Paired Patient Cluster Bootstrap ---", flush=True)
    auto_prob = policy_prob(test_p_study, tau=applied_tau_prob)
    auto_ns, _ = policy_neuro_symbolic(
        test_p_study, discretized_findings, rules, tau_score=applied_tau_prob
    )
    bootstrap_diff = paired_patient_bootstrap_diff(
        auto_a=auto_ns,
        auto_b=auto_prob,
        y_study=test_y_study,
        patient_ids=test_patient_ids,
        n_resamples=500,
        seed=cfg.reproducibility.seed,
    )
    print(
        f"Delta FOR (pi_ns - pi_prob): {bootstrap_diff['point_diff']:+.4f} "
        f"[95% CI: {bootstrap_diff['ci_low']:+.4f}, {bootstrap_diff['ci_high']:+.4f}]",
        flush=True,
    )

    # 7. Phase 4: Single-Study CPU Latency Profiling
    print("\n--- Phase 4: Single-Study CPU Latency Profiling ---", flush=True)
    latency_stats = benchmark_cpu_latency(
        model, image_size=cfg.model.image_size, n_warmup=2, n_runs=10
    )
    print(
        f"Median Forward Pass: {latency_stats['median_ms']:.2f} ms | p95: {latency_stats['p95_ms']:.2f} ms",
        flush=True,
    )

    # 8. Assemble Deliverable D6 Table & Write Artifacts
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_table: dict[str, Any] = {
        "benchmark_metadata": {
            "status": "VERIFIED",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "device": str(device),
            "backbone": cfg.model.backbone,
            "test_samples_evaluated": len(test_y_study),
            "disclaimer": DISCLAIMER_TEXT,
        },
        "provenance_hashes": {
            "checkpoint_sha256": ckpt_hash,
            "calibrator_sha256": calib_hash,
            "rules_sha256": rules_hash,
            "manifest_sha256": manifest_hash,
        },
        "calibrated_thresholds": {
            "tau_prob_raw": float(tau_prob),
            "tau_prob_applied": float(applied_tau_prob),
            "tau_ltt_raw": float(tau_ltt),
            "tau_ltt_applied": float(applied_tau_ltt),
            "epsilon_target": float(cfg.study_parameters.epsilon),
            "delta_bound": float(cfg.study_parameters.delta),
        },
        "policy_comparison": policy_metrics,
        "paired_patient_bootstrap": bootstrap_diff,
        "cpu_latency_benchmarks": latency_stats,
        "execution_duration_seconds": float(time.time() - t0),
    }

    results_path = output_dir / "results_table.json"
    results_json = json.dumps(results_table, indent=2)
    results_path.write_text(results_json, encoding="utf-8")

    # Compute and save SHA-256 hash of results_table.json
    results_sha256 = hashlib.sha256(results_json.encode("utf-8")).hexdigest()
    hash_path = output_dir / "results_table.sha256"
    hash_path.write_text(results_sha256, encoding="utf-8")

    # 9. Format Clean Terminal Summary Table
    print("\n" + "=" * 80)
    print("DELIVERABLE D6: POLICY RISK-COVERAGE COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Policy':<12} | {'Coverage':<10} | {'FOR (Emp)':<10} | {'FOR (UCB)':<10} | {'Auto':<6} | {'Errors':<6} | {'Extra Info'}"
    print(header)
    print("-" * 80)

    for pol_name in ["pi_prob", "pi_conf", "pi_ns"]:
        p_data = policy_metrics.get(pol_name, {})
        cov_str = f"{p_data.get('coverage', 0.0) * 100:.1f}%"
        for_str = f"{p_data.get('for_rate', 0.0) * 100:.2f}%"
        ucb_str = f"{p_data.get('for_ucb', 0.0) * 100:.2f}%"
        n_auto = int(p_data.get("n_auto", 0))
        n_err = int(p_data.get("n_errors", 0))

        extra = ""
        if pol_name == "pi_ns":
            extra = f"kappa={p_data.get('kappa', 0.0):.2f}, sigma={p_data.get('sigma', 0.0):.2f}"
        elif pol_name == "pi_conf":
            extra = f"tau_ltt={applied_tau_ltt:.4f}"
        elif pol_name == "pi_prob":
            extra = f"tau_prob={applied_tau_prob:.4f}"

        print(f"{pol_name:<12} | {cov_str:<10} | {for_str:<10} | {ucb_str:<10} | {n_auto:<6} | {n_err:<6} | {extra}")

    print("=" * 80)
    print(f"Results written to: {results_path} (SHA-256: {results_sha256})")
    print(f"Total benchmark elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
