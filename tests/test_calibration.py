"""Unit tests for temperature scaling, study-level calibration, and Clopper-Pearson risk bounding."""

from pathlib import Path

import numpy as np
import pytest

from cxr_triage.calibration.calibrator import (
    StudyCalibrator,
    clopper_pearson_ucb,
    compute_study_score,
)


def test_compute_study_score_properties() -> None:
    """Verify noisy-OR mathematical pooling properties."""
    # 1. All zero input -> 0.0
    zero_probs = np.zeros((3, 14))
    np.testing.assert_allclose(compute_study_score(zero_probs), [0.0, 0.0, 0.0])

    # 2. Single finding non-zero -> matches exactly that finding
    single_prob = np.zeros((1, 14))
    single_prob[0, 2] = 0.82
    np.testing.assert_allclose(compute_study_score(single_prob), [0.82], atol=1e-6)

    # 3. Two independent findings at 0.5 -> 1 - (1 - 0.5)*(1 - 0.5) = 0.75
    two_probs = np.zeros((1, 14))
    two_probs[0, 0] = 0.5
    two_probs[0, 1] = 0.5
    np.testing.assert_allclose(compute_study_score(two_probs), [0.75], atol=1e-6)

    # 4. Any finding = 1.0 -> study score = 1.0
    one_prob = np.zeros((1, 14))
    one_prob[0, 5] = 1.0
    np.testing.assert_allclose(compute_study_score(one_prob), [1.0])


def test_clopper_pearson_ucb_exact_and_bounds() -> None:
    """Verify Clopper-Pearson exact UCB values and theoretical boundaries."""
    # Boundary: n = 0 -> 1.0
    assert clopper_pearson_ucb(k_errors=0, n_total=0, delta=0.05) == 1.0

    # Boundary: k = n -> 1.0
    assert clopper_pearson_ucb(k_errors=50, n_total=50, delta=0.05) == 1.0

    # Theoretical exact check: k=0, n=100, delta=0.05
    # 1 - delta^(1/n) = 1 - 0.05^0.01 ≈ 0.029512
    ucb_k0 = clopper_pearson_ucb(k_errors=0, n_total=100, delta=0.05)
    assert ucb_k0 == pytest.approx(0.029512, abs=1e-4)

    # Monotonicity in k: increasing k must increase UCB
    ucb_k1 = clopper_pearson_ucb(k_errors=1, n_total=100, delta=0.05)
    ucb_k2 = clopper_pearson_ucb(k_errors=2, n_total=100, delta=0.05)
    assert ucb_k0 < ucb_k1 < ucb_k2

    # Monotonicity in n: increasing n for fixed k must decrease UCB
    ucb_n200 = clopper_pearson_ucb(k_errors=1, n_total=200, delta=0.05)
    assert ucb_n200 < ucb_k1

    # Exception checks
    with pytest.raises(ValueError, match="delta must be strictly in"):
        clopper_pearson_ucb(k_errors=1, n_total=10, delta=0.0)

    with pytest.raises(ValueError, match="delta must be strictly in"):
        clopper_pearson_ucb(k_errors=1, n_total=10, delta=1.0)

    with pytest.raises(ValueError, match="Invalid binomial counts"):
        clopper_pearson_ucb(k_errors=-1, n_total=10, delta=0.05)

    with pytest.raises(ValueError, match="Invalid binomial counts"):
        clopper_pearson_ucb(k_errors=15, n_total=10, delta=0.05)


def test_study_calibrator_optimization_and_persistence(tmp_path: Path) -> None:
    """Verify StudyCalibrator temperature scaling, logistic study mapping, and serialization."""
    rng = np.random.default_rng(42)
    n_samples = 1500
    n_classes = 14

    # Generate synthetic data with known underlying generative logits
    base_logits = rng.normal(0, 1.5, size=(n_samples, n_classes))
    true_probs = 1.0 / (1.0 + np.exp(-base_logits))
    labels = (rng.uniform(size=(n_samples, n_classes)) < true_probs).astype(np.float32)

    # Overconfident model scales up logits by 2.5x
    overconfident_logits = base_logits * 2.5
    y_study = (labels.sum(axis=1) > 0).astype(np.int64)

    # Initial NLL of overconfident predictions
    p_uncal = 1.0 / (1.0 + np.exp(-overconfident_logits))
    nll_uncal = -float(np.mean(labels * np.log(p_uncal + 1e-12) + (1 - labels) * np.log(1 - p_uncal + 1e-12)))

    calibrator = StudyCalibrator()
    calibrator.fit(overconfident_logits, labels, y_study)

    # 1. Temperature should be fitted > 1.0 to soften overconfident logits (close to 2.5)
    assert calibrator.temperature > 1.5, f"Expected T > 1.5 for overconfident logits, got {calibrator.temperature}"
    assert calibrator.study_scale > 0.0

    # 2. Prediction outputs
    p_findings, p_study = calibrator.predict_calibrated(overconfident_logits)
    assert p_findings.shape == (n_samples, n_classes)
    assert p_study.shape == (n_samples,)
    assert (p_findings >= 0.0).all() and (p_findings <= 1.0).all()
    assert (p_study >= 0.0).all() and (p_study <= 1.0).all()

    # Calibrated NLL should be strictly lower than uncalibrated NLL
    nll_cal = -float(np.mean(labels * np.log(p_findings + 1e-12) + (1 - labels) * np.log(1 - p_findings + 1e-12)))
    assert nll_cal < nll_uncal

    # 3. Serialization and round-trip verification
    save_path = tmp_path / "calibrator.json"
    sha256_hash = calibrator.save(save_path)

    assert save_path.exists()
    assert len(sha256_hash) == 64

    loaded_calibrator = StudyCalibrator.load(save_path)
    assert loaded_calibrator.temperature == calibrator.temperature
    assert loaded_calibrator.study_scale == calibrator.study_scale
    assert loaded_calibrator.study_bias == calibrator.study_bias

    # Confirm bit-for-bit identical predictions
    p_f_loaded, p_s_loaded = loaded_calibrator.predict_calibrated(overconfident_logits)
    np.testing.assert_array_equal(p_findings, p_f_loaded)
    np.testing.assert_array_equal(p_study, p_s_loaded)
