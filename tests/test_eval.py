"""Unit and property tests for evaluation benchmarking engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cxr_triage.config import NIH_CHESTXRAY14_LABELS
from cxr_triage.eval.benchmark import (
    benchmark_cpu_latency,
    evaluate_all_policies,
    paired_patient_bootstrap_diff,
)
from cxr_triage.models.triage_net import TriageNet
from cxr_triage.rules.engine import load_rule_registry


@pytest.fixture
def loaded_rules() -> tuple:
    """Fixture loading the frozen clinical rule registry."""
    rules_path = Path("configs/rules-v1.0.json")
    rules, _ = load_rule_registry(rules_path, frozenset(NIH_CHESTXRAY14_LABELS))
    return rules


def test_paired_patient_bootstrap_diff_properties() -> None:
    """Test paired patient bootstrap diff determinism, zero diff, and CI ordering."""
    rng = np.random.default_rng(100)
    n_samples = 200
    patient_ids = rng.choice(50, size=n_samples)
    y_study = rng.binomial(1, 0.3, size=n_samples)

    # 1. Identical decisions should yield exact zero diff and tight zero CI
    auto_same = rng.binomial(1, 0.5, size=n_samples).astype(bool)
    res_identical = paired_patient_bootstrap_diff(
        auto_a=auto_same,
        auto_b=auto_same,
        y_study=y_study,
        patient_ids=patient_ids,
        n_resamples=100,
        seed=42,
    )
    assert res_identical["point_diff"] == 0.0
    assert res_identical["ci_low"] == 0.0
    assert res_identical["ci_high"] == 0.0

    # 2. Differing decisions: check determinism and interval monotonicity
    auto_a = rng.binomial(1, 0.6, size=n_samples).astype(bool)
    auto_b = rng.binomial(1, 0.4, size=n_samples).astype(bool)

    res_1 = paired_patient_bootstrap_diff(
        auto_a=auto_a,
        auto_b=auto_b,
        y_study=y_study,
        patient_ids=patient_ids,
        n_resamples=200,
        seed=42,
    )
    res_2 = paired_patient_bootstrap_diff(
        auto_a=auto_a,
        auto_b=auto_b,
        y_study=y_study,
        patient_ids=patient_ids,
        n_resamples=200,
        seed=42,
    )

    # Exact reproducibility with identical seed
    assert res_1["point_diff"] == res_2["point_diff"]
    assert res_1["ci_low"] == res_2["ci_low"]
    assert res_1["ci_high"] == res_2["ci_high"]

    # CI ordering
    assert res_1["ci_low"] <= res_1["point_diff"] <= res_1["ci_high"]


def test_benchmark_cpu_latency() -> None:
    """Test single-study CPU forward latency profiling."""
    model = TriageNet(
        backbone_name="mobilevit_xxs",
        num_classes=14,
        pretrained=False,
    )

    stats = benchmark_cpu_latency(model, image_size=224, n_warmup=2, n_runs=5)

    assert "median_ms" in stats
    assert "p95_ms" in stats
    assert "mean_ms" in stats
    assert "std_ms" in stats

    assert stats["median_ms"] > 0.0
    assert stats["p95_ms"] > 0.0
    assert stats["p95_ms"] >= stats["median_ms"]


def test_evaluate_all_policies(loaded_rules: tuple) -> None:
    """Test comparative policy evaluation output matches Deliverable D6 schema."""
    rng = np.random.default_rng(200)
    n_samples = 40

    p_study = rng.uniform(0.01, 0.8, size=n_samples)
    y_study = (p_study > 0.4).astype(int)
    discretized = [{label: "absent" for label in NIH_CHESTXRAY14_LABELS} for _ in range(n_samples)]

    # Ensemble predictions (3 models x 40 samples)
    ensemble_p = np.stack([
        p_study,
        np.clip(p_study + rng.normal(0, 0.02, size=n_samples), 0, 1),
        np.clip(p_study + rng.normal(0, 0.02, size=n_samples), 0, 1),
    ])

    results = evaluate_all_policies(
        p_study=p_study,
        y_study=y_study,
        tau_prob=0.3,
        tau_ltt=0.25,
        rules=loaded_rules,
        discretized_findings=discretized,
        ensemble_p_study=ensemble_p,
        tau_var=0.01,
        delta=0.05,
    )

    expected_policies = {"pi_prob", "pi_conf", "pi_ns", "pi_var"}
    assert expected_policies.issubset(results.keys())

    for pol in expected_policies:
        m = results[pol]
        assert 0.0 <= m["coverage"] <= 1.0
        assert 0.0 <= m["for_rate"] <= 1.0
        assert 0.0 <= m["for_ucb"] <= 1.0
        assert m["n_total"] == float(n_samples)
        assert m["n_auto"] <= m["n_total"]
        assert m["n_errors"] <= m["n_auto"]

    # pi_ns should include clinical rule metrics kappa and sigma
    assert "kappa" in results["pi_ns"]
    assert "sigma" in results["pi_ns"]
    assert 0.0 <= results["pi_ns"]["kappa"] <= 1.0
    assert 0.0 <= results["pi_ns"]["sigma"] <= 1.0
