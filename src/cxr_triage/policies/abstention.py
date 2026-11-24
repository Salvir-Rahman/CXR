"""Four de-confounded abstention policies and study-level risk-coverage engine.

Implements:
    - pi_prob: Posterior risk thresholding
    - pi_var: Deep ensemble disagreement variance thresholding
    - pi_conf: Conformal risk-bounded thresholding via Patient-Cluster Learn-Then-Test (LTT)
    - pi_ns: Neuro-symbolic triage with clinical rule violation and severity entropy vetoes
    - Shared upstream image quality gating to eliminate policy confounding
    - Exact finite-sample Clopper-Pearson risk bounding and policy metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from cxr_triage.calibration.calibrator import clopper_pearson_ucb
from cxr_triage.rules.engine import (
    SymbolicRule,
    compute_normalized_entropy,
    evaluate_rules,
)

Decision = Literal["auto", "esc"]


@dataclass(frozen=True)
class PolicyMetrics:
    """Evaluation metrics for an automated triage abstention policy.

    Attributes:
        coverage: Fraction of all studies auto-reported (n_auto / n_total).
        for_rate: Empirical False Omission Rate (n_errors / n_auto).
        for_ucb: Exact one-sided (1 - delta) Clopper-Pearson upper confidence bound on FOR.
        n_auto: Number of studies auto-reported.
        n_errors: Number of auto-reported studies that are abnormal (y_study == 1).
        n_total: Total number of evaluated studies.
    """

    coverage: float
    for_rate: float
    for_ucb: float
    n_auto: int
    n_errors: int
    n_total: int


def apply_quality_gate(
    decisions: np.ndarray, is_adequate_quality: np.ndarray | None
) -> np.ndarray:
    """Apply upstream image quality pre-filter identically across all policies.

    Any study deemed inadequate quality is forced to escalate ('esc' / False),
    ensuring that image-degradation effects do not confound comparative policy evaluation.

    Args:
        decisions: Boolean array where True represents 'auto' and False represents 'esc'.
        is_adequate_quality: Optional boolean array where True indicates diagnostic quality.

    Returns:
        Boolean array with forced escalation on inadequate quality studies.
    """
    gated = np.asarray(decisions, dtype=bool).copy()
    if is_adequate_quality is not None:
        quality_mask = np.asarray(is_adequate_quality, dtype=bool)
        gated = gated & quality_mask
    return gated


def fit_clopper_pearson_threshold(
    scores: np.ndarray,
    y_study: np.ndarray,
    epsilon: float,
    delta: float,
    grid_size: int = 200,
    is_adequate_quality: np.ndarray | None = None,
) -> float:
    r"""Find the largest risk score threshold tau guaranteeing Clopper-Pearson FOR_UCB <= epsilon.

    Searches a candidate grid \tau \in [0, 1] on calibration data such that among auto-reported
    studies (score <= \tau and adequate quality), the exact Clopper-Pearson upper bound satisfies:
        \mathrm{FOR}_{\mathcal{S}}^{\mathrm{UCB}} \le \epsilon

    Args:
        scores: Calibrated study abnormality probabilities \hat{p}_{\mathcal{S}} in [0, 1].
        y_study: Ground truth binary study labels (1 = abnormal, 0 = normal).
        epsilon: Target upper bound on False Omission Rate.
        delta: Error probability bound for (1 - delta) confidence.
        grid_size: Number of threshold evaluation points in [0, 1].
        is_adequate_quality: Optional boolean mask of diagnostic image quality.

    Returns:
        Largest valid threshold tau in [0, 1], or 0.0 if no threshold satisfies the bound.
    """
    scores_arr = np.asarray(scores, dtype=float)
    y_arr = np.asarray(y_study, dtype=int)
    candidates = np.linspace(0.0, 1.0, grid_size)

    valid_taus: list[float] = []
    for tau in candidates:
        auto_mask = scores_arr <= tau
        auto_mask = apply_quality_gate(auto_mask, is_adequate_quality)
        n_auto = int(np.sum(auto_mask))
        if n_auto == 0:
            continue
        n_errors = int(np.sum(auto_mask & (y_arr == 1)))
        ucb = clopper_pearson_ucb(n_errors, n_auto, delta=delta)
        if ucb <= epsilon:
            valid_taus.append(float(tau))

    return max(valid_taus) if valid_taus else 0.0


def fit_cluster_ltt_threshold(
    scores: np.ndarray,
    y_study: np.ndarray,
    patient_ids: np.ndarray,
    epsilon: float,
    delta: float,
    grid_size: int = 50,
    is_adequate_quality: np.ndarray | None = None,
) -> float:
    r"""Fit risk threshold tau_ltt via Patient-Cluster Learn-Then-Test (LTT) for pi_conf.

    Guarantees patient-clustered risk control under clustered non-IID sampling by testing
    each candidate threshold with Bonferroni-corrected significance level \delta / M.
    A patient cluster is counted as an error if ANY study auto-reported for that patient has y_study == 1.

    Args:
        scores: Calibrated study abnormality probabilities in [0, 1].
        y_study: Ground truth binary study labels.
        patient_ids: Patient cluster identifier for each study.
        epsilon: Target risk tolerance bound.
        delta: Family-wise error rate bound.
        grid_size: Number of candidate thresholds (M) tested in parallel.
        is_adequate_quality: Optional boolean mask of diagnostic image quality.

    Returns:
        Largest threshold tau_ltt satisfying the cluster-level risk bound, or 0.0.
    """
    scores_arr = np.asarray(scores, dtype=float)
    y_arr = np.asarray(y_study, dtype=int)
    patients_arr = np.asarray(patient_ids)

    unique_patients = np.unique(patients_arr)
    candidates = np.linspace(0.0, 1.0, grid_size)
    bonferroni_delta = delta / grid_size

    # Pre-index patient study locations for sub-millisecond evaluation
    patient_indices = [np.where(patients_arr == p)[0] for p in unique_patients]

    valid_taus: list[float] = []
    for tau in candidates:
        study_auto = apply_quality_gate(scores_arr <= tau, is_adequate_quality)

        n_auto_patients = 0
        n_error_patients = 0

        for idxs in patient_indices:
            pat_auto = study_auto[idxs]
            if np.any(pat_auto):
                n_auto_patients += 1
                pat_y = y_arr[idxs]
                if np.any(pat_auto & (pat_y == 1)):
                    n_error_patients += 1

        if n_auto_patients == 0:
            continue

        ucb = clopper_pearson_ucb(n_error_patients, n_auto_patients, delta=bonferroni_delta)
        if ucb <= epsilon:
            valid_taus.append(float(tau))

    return max(valid_taus) if valid_taus else 0.0


def policy_prob(
    p_study: np.ndarray, tau: float, is_adequate_quality: np.ndarray | None = None
) -> np.ndarray:
    r"""Abstention policy pi_prob: posterior probability thresholding.

    Auto-reports if \hat{p}_{\mathcal{S}}(x) <= tau and image quality is adequate.

    Args:
        p_study: Calibrated study abnormality probabilities of shape (N,).
        tau: Scalar decision threshold.
        is_adequate_quality: Optional quality mask of shape (N,).

    Returns:
        Boolean array of shape (N,) where True indicates 'auto' and False indicates 'esc'.
    """
    p_arr = np.asarray(p_study, dtype=float)
    raw_decisions = p_arr <= tau
    return apply_quality_gate(raw_decisions, is_adequate_quality)


def policy_disagreement(
    ensemble_p_study: np.ndarray,
    tau_var: float,
    is_adequate_quality: np.ndarray | None = None,
) -> np.ndarray:
    r"""Abstention policy pi_var: ensemble predictive variance thresholding.

    Auto-reports if the variance across ensemble model predictions is <= tau_var
    and image quality is adequate.

    Args:
        ensemble_p_study: Array of shape (M, N) containing study probabilities from M models.
        tau_var: Scalar predictive variance threshold.
        is_adequate_quality: Optional quality mask of shape (N,).

    Returns:
        Boolean array of shape (N,) where True indicates 'auto' and False indicates 'esc'.
    """
    ens_arr = np.asarray(ensemble_p_study, dtype=float)
    if ens_arr.ndim != 2:
        raise ValueError(f"ensemble_p_study must be 2D (M, N), got shape {ens_arr.shape}")
    variances = np.var(ens_arr, axis=0, ddof=0)
    raw_decisions = variances <= tau_var
    return apply_quality_gate(raw_decisions, is_adequate_quality)


def policy_conformal(
    p_study: np.ndarray, tau_ltt: float, is_adequate_quality: np.ndarray | None = None
) -> np.ndarray:
    r"""Abstention policy pi_conf: finite-sample conformal risk-bounded triage.

    Auto-reports if \hat{p}_{\mathcal{S}}(x) <= tau_ltt and image quality is adequate.

    Args:
        p_study: Calibrated study abnormality probabilities of shape (N,).
        tau_ltt: Conformal threshold calibrated via Learn-Then-Test.
        is_adequate_quality: Optional quality mask of shape (N,).

    Returns:
        Boolean array of shape (N,) where True indicates 'auto' and False indicates 'esc'.
    """
    return policy_prob(p_study, tau=tau_ltt, is_adequate_quality=is_adequate_quality)


def policy_neuro_symbolic(
    p_study: np.ndarray,
    discretized_findings: list[dict[str, str]],
    rules: tuple[SymbolicRule, ...],
    tau_score: float,
    struct_probs: np.ndarray | None = None,
    tau_entropy: float | None = None,
    use_entropy_veto: bool = False,
    is_adequate_quality: np.ndarray | None = None,
    y_study: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    r"""Abstention policy pi_ns: de-confounded neuro-symbolic triage with dual vetoes.

    A study is auto-reported if and only if ALL of the following criteria hold:
        1. Posterior risk condition: \hat{p}_{\mathcal{S}}(x) <= tau_score
        2. Symbolic consistency veto: No auditable Horn clause is violated (|V| == 0)
        3. Severity entropy veto (if enabled): \max_k H_k(x) <= tau_entropy
        4. Image quality condition: is_adequate_quality is True (if provided)

    Args:
        p_study: Calibrated study abnormality probabilities of shape (N,).
        discretized_findings: List of N pathology discretization dictionaries.
        rules: Tuple of auditable SymbolicRule Horn clauses.
        tau_score: Calibrated risk score threshold.
        struct_probs: Optional 3-state severity probabilities of shape (N, 14, 3).
        tau_entropy: Optional normalized entropy threshold in [0, 1].
        use_entropy_veto: Whether to enforce the severity entropy veto.
        is_adequate_quality: Optional quality mask of shape (N,).
        y_study: Optional ground truth binary labels to compute rule metrics (kappa, sigma).

    Returns:
        Tuple of (auto_decisions_mask, metrics_dict).
    """
    p_arr = np.asarray(p_study, dtype=float)
    n = len(p_arr)
    if len(discretized_findings) != n:
        raise ValueError(
            f"Length mismatch: p_study has {n} items, discretized_findings has {len(discretized_findings)}"
        )

    auto_mask = p_arr <= tau_score

    # 1. Symbolic clinical rule consistency veto
    for i in range(n):
        if not auto_mask[i]:
            continue
        eval_res = evaluate_rules(discretized_findings[i], rules)
        if eval_res.is_violated:
            auto_mask[i] = False

    # 2. Severity distribution entropy veto
    if use_entropy_veto and struct_probs is not None and tau_entropy is not None:
        struct_arr = np.asarray(struct_probs, dtype=float)
        # Compute normalized entropy for each finding: shape (N, 14)
        entropies = compute_normalized_entropy(struct_arr)
        max_entropies = np.max(entropies, axis=-1)  # shape (N,)
        entropy_veto = max_entropies > tau_entropy
        auto_mask = auto_mask & (~entropy_veto)

    # 3. Upstream image quality gate
    auto_mask = apply_quality_gate(auto_mask, is_adequate_quality)

    # Compute rule coverage (kappa) and rule suppression (sigma) if ground truth is supplied
    metrics: dict[str, float] = {}
    if y_study is not None:
        metrics = compute_rule_metrics(discretized_findings, rules, y_study)
    else:
        n_violations = sum(
            1 for f in discretized_findings if evaluate_rules(f, rules).is_violated
        )
        metrics["violation_rate"] = float(n_violations / n) if n > 0 else 0.0

    return auto_mask, metrics


def compute_rule_metrics(
    discretized_findings: list[dict[str, str]],
    rules: tuple[SymbolicRule, ...],
    y_study: np.ndarray,
) -> dict[str, float]:
    r"""Compute clinical rule coverage (kappa) and rule suppression (sigma).

    Formulas:
        \kappa = \mathbb{P}(|\mathcal{V}| > 0 \mid Y_{\mathcal{S}} = 1)  (Coverage: abnormal studies with violations)
        \sigma = \mathbb{P}(|\mathcal{V}| > 0 \mid Y_{\mathcal{S}} = 0)  (Suppression: normal studies with violations)

    Args:
        discretized_findings: List of N pathology state dictionaries.
        rules: Tuple of auditable SymbolicRule Horn clauses.
        y_study: Binary ground-truth study labels of shape (N,).

    Returns:
        Dictionary containing 'rule_coverage_kappa', 'rule_suppression_sigma', 'kappa', and 'sigma'.
    """
    y_arr = np.asarray(y_study, dtype=int)
    n = len(discretized_findings)
    if len(y_arr) != n:
        raise ValueError(
            f"Length mismatch: discretized_findings has {n} items, y_study has {len(y_arr)}"
        )

    has_violation = np.zeros(n, dtype=bool)
    for i, findings in enumerate(discretized_findings):
        eval_res = evaluate_rules(findings, rules)
        if eval_res.is_violated:
            has_violation[i] = True

    abnormal_mask = y_arr == 1
    normal_mask = y_arr == 0

    n_abnormal = int(np.sum(abnormal_mask))
    n_normal = int(np.sum(normal_mask))

    kappa = (
        float(np.sum(has_violation & abnormal_mask) / n_abnormal)
        if n_abnormal > 0
        else 0.0
    )
    sigma = (
        float(np.sum(has_violation & normal_mask) / n_normal)
        if n_normal > 0
        else 0.0
    )

    return {
        "rule_coverage_kappa": kappa,
        "rule_suppression_sigma": sigma,
        "kappa": kappa,
        "sigma": sigma,
    }


def evaluate_policy_metrics(
    auto_mask: np.ndarray, y_study: np.ndarray, delta: float = 0.05
) -> PolicyMetrics:
    """Compute empirical coverage, FOR, and exact Clopper-Pearson FOR_UCB.

    Args:
        auto_mask: Boolean array where True indicates 'auto' and False indicates 'esc'.
        y_study: Binary ground truth study labels of shape (N,).
        delta: Error probability bound for Clopper-Pearson UCB.

    Returns:
        PolicyMetrics dataclass instance.
    """
    auto_arr = np.asarray(auto_mask, dtype=bool)
    y_arr = np.asarray(y_study, dtype=int)

    n_total = len(auto_arr)
    n_auto = int(np.sum(auto_arr))
    n_errors = int(np.sum(auto_arr & (y_arr == 1)))

    coverage = float(n_auto / n_total) if n_total > 0 else 0.0
    for_rate = float(n_errors / n_auto) if n_auto > 0 else 0.0
    for_ucb = clopper_pearson_ucb(n_errors, n_auto, delta=delta)

    return PolicyMetrics(
        coverage=coverage,
        for_rate=for_rate,
        for_ucb=for_ucb,
        n_auto=n_auto,
        n_errors=n_errors,
        n_total=n_total,
    )
