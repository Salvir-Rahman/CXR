"""Tests for de-confounded abstention policies and study-level risk-coverage engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cxr_triage.config import NIH_CHESTXRAY14_LABELS
from cxr_triage.policies.abstention import (
    compute_rule_metrics,
    evaluate_policy_metrics,
    fit_clopper_pearson_threshold,
    fit_cluster_ltt_threshold,
    policy_conformal,
    policy_disagreement,
    policy_neuro_symbolic,
    policy_prob,
)
from cxr_triage.rules.engine import load_rule_registry


@pytest.fixture
def loaded_rules() -> tuple:
    """Fixture loading the frozen clinical rule registry."""
    rules_path = Path("configs/rules-v1.0.json")
    rules, _ = load_rule_registry(rules_path, frozenset(NIH_CHESTXRAY14_LABELS))
    return rules


def test_quality_gate_forces_escalation_across_all_policies(loaded_rules: tuple) -> None:
    """Test that upstream quality gate forces escalation identically across all 4 policies."""
    n_samples = 10
    # All scores are zero (normally auto-reported under any positive threshold)
    p_study = np.zeros(n_samples)
    ensemble_p = np.zeros((3, n_samples))
    discretized = [{label: "absent" for label in NIH_CHESTXRAY14_LABELS} for _ in range(n_samples)]

    # Mark samples 3 and 7 as compromised image quality
    is_adequate = np.ones(n_samples, dtype=bool)
    is_adequate[3] = False
    is_adequate[7] = False

    dec_prob = policy_prob(p_study, tau=0.5, is_adequate_quality=is_adequate)
    dec_var = policy_disagreement(ensemble_p, tau_var=0.01, is_adequate_quality=is_adequate)
    dec_conf = policy_conformal(p_study, tau_ltt=0.5, is_adequate_quality=is_adequate)
    dec_ns, _ = policy_neuro_symbolic(
        p_study, discretized, loaded_rules, tau_score=0.5, is_adequate_quality=is_adequate
    )

    for dec in [dec_prob, dec_var, dec_conf, dec_ns]:
        assert dec[3] is False or dec[3] == np.bool_(False)
        assert dec[7] is False or dec[7] == np.bool_(False)
        # Other samples should remain True (auto)
        assert np.all(dec[[0, 1, 2, 4, 5, 6, 8, 9]])


def test_fit_clopper_pearson_threshold_guarantees_risk_bound() -> None:
    """Test that fit_clopper_pearson_threshold yields FOR_UCB <= epsilon on calibration data."""
    rng = np.random.default_rng(42)
    n_samples = 500

    # Synthetic calibration risk scores and true labels
    scores = rng.beta(0.5, 2.0, size=n_samples)
    # Probability of abnormality increases monotonically with score
    y_study = (rng.uniform(0.0, 1.0, size=n_samples) < scores).astype(int)

    epsilon = 0.05
    delta = 0.05
    tau_opt = fit_clopper_pearson_threshold(scores, y_study, epsilon=epsilon, delta=delta)

    assert 0.0 <= tau_opt <= 1.0

    if tau_opt > 0.0:
        auto_mask = scores <= tau_opt
        metrics = evaluate_policy_metrics(auto_mask, y_study, delta=delta)
        assert metrics.for_ucb <= epsilon
        assert metrics.n_auto > 0


def test_fit_clopper_pearson_threshold_unachievable_epsilon() -> None:
    """Test that fit_clopper_pearson_threshold returns 0.0 when target risk is impossible."""
    scores = np.array([0.1, 0.2, 0.3])
    y_study = np.array([1, 1, 1])  # All errors

    tau_opt = fit_clopper_pearson_threshold(scores, y_study, epsilon=0.01, delta=0.05)
    assert tau_opt == 0.0


def test_fit_cluster_ltt_threshold_patient_clustering() -> None:
    """Test Patient-Cluster Learn-Then-Test threshold fitting on multi-study patient clusters."""
    rng = np.random.default_rng(123)
    n_studies = 300

    # 100 unique patients with ~3 studies each
    patient_ids = rng.choice(100, size=n_studies)
    scores = rng.beta(0.5, 3.0, size=n_studies)
    y_study = (scores > 0.6).astype(int)

    epsilon = 0.10
    delta = 0.05

    tau_ltt = fit_cluster_ltt_threshold(
        scores, y_study, patient_ids, epsilon=epsilon, delta=delta, grid_size=30
    )

    assert 0.0 <= tau_ltt <= 1.0
    auto_mask = policy_conformal(scores, tau_ltt=tau_ltt)
    assert isinstance(auto_mask, np.ndarray)
    assert auto_mask.dtype == bool


def test_policy_disagreement_with_three_member_ensemble() -> None:
    """Test policy_disagreement with a 3-member ensemble of shape (3, N)."""
    # 5 agreeing samples (identical predictions) and 5 disagreeing samples
    agreeing = np.array([[0.05, 0.05, 0.05, 0.05, 0.05],
                         [0.05, 0.05, 0.05, 0.05, 0.05],
                         [0.05, 0.05, 0.05, 0.05, 0.05]])
    disagreeing = np.array([[0.01, 0.02, 0.05, 0.10, 0.20],
                            [0.50, 0.40, 0.60, 0.70, 0.80],
                            [0.90, 0.80, 0.85, 0.95, 0.90]])

    ensemble_p = np.concatenate([agreeing, disagreeing], axis=1)  # shape (3, 10)
    assert ensemble_p.shape == (3, 10)

    # Variance threshold 0.001 should auto-report first 5 and escalate last 5
    decisions = policy_disagreement(ensemble_p, tau_var=0.001)
    assert np.all(decisions[:5])
    assert not np.any(decisions[5:])

    # Non-2D input should raise ValueError
    with pytest.raises(ValueError, match="ensemble_p_study must be 2D"):
        policy_disagreement(np.zeros((10,)), tau_var=0.01)


def test_policy_neuro_symbolic_conservatism_and_veto(loaded_rules: tuple) -> None:
    """Test that policy_neuro_symbolic is strictly conservative relative to policy_prob."""
    # All samples have low predicted probability
    p_study = np.array([0.05, 0.05, 0.05, 0.05])
    tau = 0.10

    # Sample 0: completely normal
    f0 = {label: "absent" for label in NIH_CHESTXRAY14_LABELS}

    # Sample 1: Violates R1.2 (Consolidation=present, Infiltration=absent)
    f1 = {label: "absent" for label in NIH_CHESTXRAY14_LABELS}
    f1["Consolidation"] = "present"
    f1["Infiltration"] = "absent"

    # Sample 2: normal findings but high entropy
    f2 = {label: "absent" for label in NIH_CHESTXRAY14_LABELS}

    # Sample 3: normal findings, low entropy
    f3 = {label: "absent" for label in NIH_CHESTXRAY14_LABELS}

    discretized = [f0, f1, f2, f3]

    # Struct probs: shape (4, 14, 3)
    # Low entropy for sample 0, 1, 3 (one-hot [1, 0, 0])
    struct_probs = np.zeros((4, 14, 3))
    struct_probs[:, :, 0] = 1.0
    # Uniform high entropy for sample 2 across all 14 findings
    struct_probs[2, :, :] = 1.0 / 3.0

    # 1. Pure symbolic veto test (use_entropy_veto=False)
    dec_prob = policy_prob(p_study, tau=tau)
    dec_ns_pure, _ = policy_neuro_symbolic(
        p_study, discretized, loaded_rules, tau_score=tau, use_entropy_veto=False
    )

    # Pure probability approves all 4
    assert np.all(dec_prob)
    # Neuro-symbolic approves 0, 2, 3 but vetoes 1
    assert dec_ns_pure[0]
    assert not dec_ns_pure[1]
    assert dec_ns_pure[2]
    assert dec_ns_pure[3]

    # Strict conservatism / subset property: auto_ns <= auto_prob everywhere
    assert np.all(dec_ns_pure <= dec_prob)

    # 2. Dual veto test (use_entropy_veto=True, tau_entropy=0.5)
    dec_ns_dual, metrics = policy_neuro_symbolic(
        p_study,
        discretized,
        loaded_rules,
        tau_score=tau,
        struct_probs=struct_probs,
        tau_entropy=0.5,
        use_entropy_veto=True,
    )

    # Dual veto vetoes Sample 1 (rule violation) and Sample 2 (entropy violation)
    assert dec_ns_dual[0]
    assert not dec_ns_dual[1]
    assert not dec_ns_dual[2]
    assert dec_ns_dual[3]

    assert "violation_rate" in metrics
    assert metrics["violation_rate"] == 0.25  # 1 out of 4 violated


def test_compute_rule_metrics_exact_kappa_and_sigma(loaded_rules: tuple) -> None:
    """Test exact computation of rule coverage kappa and rule suppression sigma."""
    # 2 abnormal studies (y_study=1), 2 normal studies (y_study=0)
    y_study = np.array([1, 1, 0, 0])

    f_clean = {label: "absent" for label in NIH_CHESTXRAY14_LABELS}
    # Violates R1.2 (Consolidation=present, Infiltration=absent)
    f_violated = {label: "absent" for label in NIH_CHESTXRAY14_LABELS}
    f_violated["Consolidation"] = "present"
    f_violated["Infiltration"] = "absent"

    # Study 0 (abnormal): violated
    # Study 1 (abnormal): clean
    # Study 2 (normal): clean
    # Study 3 (normal): violated
    discretized = [f_violated, f_clean, f_clean, f_violated]

    metrics = compute_rule_metrics(discretized, loaded_rules, y_study)

    # kappa = P(|V| > 0 | Y_S = 1) = 1 / 2 = 0.5
    assert metrics["rule_coverage_kappa"] == 0.5
    assert metrics["kappa"] == 0.5

    # sigma = P(|V| > 0 | Y_S = 0) = 1 / 2 = 0.5
    assert metrics["rule_suppression_sigma"] == 0.5
    assert metrics["sigma"] == 0.5


def test_policy_metrics_dataclass_and_edge_cases() -> None:
    """Test PolicyMetrics computation on empty and boundary inputs."""
    # Zero auto-reports
    auto_empty = np.array([False, False, False])
    y_study = np.array([0, 1, 0])
    m_empty = evaluate_policy_metrics(auto_empty, y_study)
    assert m_empty.n_auto == 0
    assert m_empty.coverage == 0.0
    assert m_empty.for_rate == 0.0
    assert m_empty.for_ucb == 1.0  # clopper_pearson_ucb returns 1.0 when n=0

    # 100% auto-reports with 0 errors
    auto_all = np.array([True, True, True, True])
    y_all_normal = np.array([0, 0, 0, 0])
    m_perfect = evaluate_policy_metrics(auto_all, y_all_normal)
    assert m_perfect.n_auto == 4
    assert m_perfect.n_errors == 0
    assert m_perfect.coverage == 1.0
    assert m_perfect.for_rate == 0.0
    assert m_perfect.for_ucb < 1.0
