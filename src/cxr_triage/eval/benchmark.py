"""Evaluation benchmark engine for comparative policy assessment and latency profiling."""

from __future__ import annotations

import time

import numpy as np
import torch

from cxr_triage.models.triage_net import TriageNet
from cxr_triage.policies.abstention import (
    evaluate_policy_metrics,
    policy_conformal,
    policy_disagreement,
    policy_neuro_symbolic,
    policy_prob,
)
from cxr_triage.rules.engine import SymbolicRule


def paired_patient_bootstrap_diff(
    auto_a: np.ndarray,
    auto_b: np.ndarray,
    y_study: np.ndarray,
    patient_ids: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    r"""Compute paired difference in False Omission Rate (Delta = FOR_A - FOR_B) with bootstrap CI.

    Resamples unique patient clusters with replacement to account for intra-patient correlation,
    evaluating the point difference and 95% empirical bootstrap confidence interval.

    Args:
        auto_a: Boolean array indicating auto-reporting decisions for Policy A (shape (N,)).
        auto_b: Boolean array indicating auto-reporting decisions for Policy B (shape (N,)).
        y_study: Ground truth binary study labels (1 = abnormal, 0 = normal) of shape (N,).
        patient_ids: Patient cluster identifier for each study of shape (N,).
        n_resamples: Number of cluster bootstrap iterations.
        seed: Random seed for reproducible bootstrap sampling.

    Returns:
        Dictionary containing 'point_diff', 'ci_low', and 'ci_high'.
    """
    auto_a_arr = np.asarray(auto_a, dtype=bool)
    auto_b_arr = np.asarray(auto_b, dtype=bool)
    y_arr = np.asarray(y_study, dtype=int)
    patient_arr = np.asarray(patient_ids)

    n_total = len(auto_a_arr)
    if not (len(auto_b_arr) == n_total and len(y_arr) == n_total and len(patient_arr) == n_total):
        raise ValueError("All inputs must have identical length.")

    # Calculate point estimate of FOR difference
    n_auto_a = int(np.sum(auto_a_arr))
    n_auto_b = int(np.sum(auto_b_arr))
    for_a = float(np.sum(auto_a_arr & (y_arr == 1)) / n_auto_a) if n_auto_a > 0 else 0.0
    for_b = float(np.sum(auto_b_arr & (y_arr == 1)) / n_auto_b) if n_auto_b > 0 else 0.0
    point_diff = for_a - for_b

    # Exact zero short-circuit if identical policies
    if np.array_equal(auto_a_arr, auto_b_arr):
        return {"point_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    unique_patients = np.unique(patient_arr)
    n_patients = len(unique_patients)
    rng = np.random.default_rng(seed)

    # Pre-index study indices per patient
    patient_indices = {p: np.where(patient_arr == p)[0] for p in unique_patients}

    diffs: list[float] = []
    for _ in range(n_resamples):
        sampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
        sampled_indices = np.concatenate([patient_indices[p] for p in sampled_patients])

        sub_auto_a = auto_a_arr[sampled_indices]
        sub_auto_b = auto_b_arr[sampled_indices]
        sub_y = y_arr[sampled_indices]

        n_sub_a = int(np.sum(sub_auto_a))
        n_sub_b = int(np.sum(sub_auto_b))

        sub_for_a = float(np.sum(sub_auto_a & (sub_y == 1)) / n_sub_a) if n_sub_a > 0 else 0.0
        sub_for_b = float(np.sum(sub_auto_b & (sub_y == 1)) / n_sub_b) if n_sub_b > 0 else 0.0
        diffs.append(sub_for_a - sub_for_b)

    diffs_arr = np.array(diffs)
    ci_low = float(np.percentile(diffs_arr, 2.5))
    ci_high = float(np.percentile(diffs_arr, 97.5))

    return {
        "point_diff": float(point_diff),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def benchmark_cpu_latency(
    model: TriageNet,
    image_size: int = 224,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> dict[str, float]:
    """Measure single-study forward pass latency on CPU in inference_only mode.

    Args:
        model: TriageNet instance.
        image_size: Input spatial dimension (H = W = image_size).
        n_warmup: Number of unmeasured warmup iterations.
        n_runs: Number of timed benchmark iterations.

    Returns:
        Dictionary containing 'median_ms', 'p95_ms', 'mean_ms', and 'std_ms'.
    """
    model.eval()
    dummy_input = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input, inference_only=True)

        durations_ms: list[float] = []
        for _ in range(n_runs):
            t_start = time.perf_counter()
            _ = model(dummy_input, inference_only=True)
            t_end = time.perf_counter()
            durations_ms.append((t_end - t_start) * 1000.0)

    durations_arr = np.array(durations_ms)
    return {
        "median_ms": float(np.median(durations_arr)),
        "p95_ms": float(np.percentile(durations_arr, 95)),
        "mean_ms": float(np.mean(durations_arr)),
        "std_ms": float(np.std(durations_arr)),
    }


def evaluate_all_policies(
    p_study: np.ndarray,
    y_study: np.ndarray,
    tau_prob: float,
    tau_ltt: float,
    rules: tuple[SymbolicRule, ...],
    discretized_findings: list[dict[str, str]],
    ensemble_p_study: np.ndarray | None = None,
    tau_var: float | None = None,
    struct_probs: np.ndarray | None = None,
    tau_entropy: float | None = None,
    use_entropy_veto: bool = False,
    is_adequate_quality: np.ndarray | None = None,
    delta: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Evaluate and benchmark all four triage policies on common calibrated predictions.

    Produces structured output matching the Deliverable D6 schema for comparative reporting.

    Args:
        p_study: Calibrated study abnormality probabilities (shape (N,)).
        y_study: Ground truth study labels (shape (N,)).
        tau_prob: Risk score threshold for pi_prob.
        tau_ltt: Conformal risk threshold for pi_conf.
        rules: Validated clinical SymbolicRule Horn clauses.
        discretized_findings: List of N pathology state dictionaries.
        ensemble_p_study: Optional multi-model predictions of shape (M, N) for pi_var.
        tau_var: Variance threshold for pi_var.
        struct_probs: Optional severity probabilities of shape (N, 14, 3).
        tau_entropy: Normalized severity entropy threshold.
        use_entropy_veto: Whether to enable the entropy veto in pi_ns.
        is_adequate_quality: Optional diagnostic image quality mask.
        delta: Error probability bound for Clopper-Pearson UCB.

    Returns:
        Dictionary mapping policy names ('pi_prob', 'pi_conf', 'pi_ns', optionally 'pi_var')
        to their respective evaluation metrics.
    """
    results: dict[str, dict[str, float]] = {}

    # 1. Evaluate pi_prob
    auto_prob = policy_prob(p_study, tau=tau_prob, is_adequate_quality=is_adequate_quality)
    m_prob = evaluate_policy_metrics(auto_prob, y_study, delta=delta)
    results["pi_prob"] = {
        "coverage": m_prob.coverage,
        "for_rate": m_prob.for_rate,
        "for_ucb": m_prob.for_ucb,
        "n_auto": float(m_prob.n_auto),
        "n_errors": float(m_prob.n_errors),
        "n_total": float(m_prob.n_total),
    }

    # 2. Evaluate pi_conf
    auto_conf = policy_conformal(p_study, tau_ltt=tau_ltt, is_adequate_quality=is_adequate_quality)
    m_conf = evaluate_policy_metrics(auto_conf, y_study, delta=delta)
    results["pi_conf"] = {
        "coverage": m_conf.coverage,
        "for_rate": m_conf.for_rate,
        "for_ucb": m_conf.for_ucb,
        "n_auto": float(m_conf.n_auto),
        "n_errors": float(m_conf.n_errors),
        "n_total": float(m_conf.n_total),
    }

    # 3. Evaluate pi_ns
    auto_ns, rule_metrics = policy_neuro_symbolic(
        p_study=p_study,
        discretized_findings=discretized_findings,
        rules=rules,
        tau_score=tau_prob,
        struct_probs=struct_probs,
        tau_entropy=tau_entropy,
        use_entropy_veto=use_entropy_veto,
        is_adequate_quality=is_adequate_quality,
        y_study=y_study,
    )
    m_ns = evaluate_policy_metrics(auto_ns, y_study, delta=delta)
    results["pi_ns"] = {
        "coverage": m_ns.coverage,
        "for_rate": m_ns.for_rate,
        "for_ucb": m_ns.for_ucb,
        "n_auto": float(m_ns.n_auto),
        "n_errors": float(m_ns.n_errors),
        "n_total": float(m_ns.n_total),
        "kappa": rule_metrics.get("rule_coverage_kappa", 0.0),
        "sigma": rule_metrics.get("rule_suppression_sigma", 0.0),
    }

    # 4. Optional evaluate pi_var
    if ensemble_p_study is not None and tau_var is not None:
        auto_var = policy_disagreement(
            ensemble_p_study, tau_var=tau_var, is_adequate_quality=is_adequate_quality
        )
        m_var = evaluate_policy_metrics(auto_var, y_study, delta=delta)
        results["pi_var"] = {
            "coverage": m_var.coverage,
            "for_rate": m_var.for_rate,
            "for_ucb": m_var.for_ucb,
            "n_auto": float(m_var.n_auto),
            "n_errors": float(m_var.n_errors),
            "n_total": float(m_var.n_total),
        }

    return results
