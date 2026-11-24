"""Post-hoc probability calibration, study-level aggregation, and finite-sample risk bounds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats  # type: ignore[import-untyped]


def compute_study_score(probs: np.ndarray) -> np.ndarray:
    r"""Compute study-level abnormality probability using the noisy-OR formulation.

    Formula:
        \hat{p}_{\mathcal{S}}(x) = 1 - \prod_{k=1}^{K} (1 - \hat{p}_k(x))

    Args:
        probs: Array of shape (..., K) containing per-finding probabilities in [0, 1].

    Returns:
        Array of shape (...) containing pooled study abnormality score in [0, 1].
    """
    clipped_probs = np.clip(probs, 0.0, 1.0)
    score = 1.0 - np.prod(1.0 - clipped_probs, axis=-1)
    return np.clip(score, 0.0, 1.0)


def clopper_pearson_ucb(k_errors: int, n_total: int, delta: float = 0.05) -> float:
    r"""Compute exact one-sided (1 - delta) Clopper-Pearson upper confidence bound on a binomial rate.

    Formula:
        \overline{R}_{\delta} = \mathrm{Beta}^{-1}(1 - \delta;\, k + 1,\, n - k)

    Args:
        k_errors: Number of observed errors (successes in binomial failure count).
        n_total: Total number of evaluated trials.
        delta: Failure probability bound in (0, 1), default 0.05 (corresponds to 95% confidence).

    Returns:
        Upper confidence bound in [0.0, 1.0].

    Raises:
        ValueError: If counts are negative, k_errors > n_total, or delta is not in (0, 1).
    """
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be strictly in (0.0, 1.0), got {delta}")

    if k_errors < 0 or n_total < 0 or k_errors > n_total:
        raise ValueError(
            f"Invalid binomial counts: k_errors={k_errors}, n_total={n_total}. "
            f"Must satisfy 0 <= k_errors <= n_total."
        )

    if n_total == 0 or k_errors == n_total:
        return 1.0

    return float(stats.beta.ppf(1.0 - delta, k_errors + 1, n_total - k_errors))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    clipped_z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped_z))


def _logit(p: np.ndarray) -> np.ndarray:
    """Numerically safe logit function with epsilon clamping."""
    p_safe = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p_safe / (1.0 - p_safe))


def _log_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable log(sigmoid(z)) = -softplus(-z)."""
    return -np.logaddexp(0.0, -z)


def _log_one_minus_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable log(1 - sigmoid(z)) = -softplus(z)."""
    return -np.logaddexp(0.0, z)


class StudyCalibrator:
    """Two-stage post-hoc calibrator for multi-label pathologies and study-level referral score."""

    def __init__(
        self,
        temperature: float = 1.0,
        study_scale: float = 1.0,
        study_bias: float = 0.0,
    ) -> None:
        """Initialize calibrator parameters."""
        self.temperature = float(temperature)
        self.study_scale = float(study_scale)
        self.study_bias = float(study_bias)

    def fit(
        self, logits: np.ndarray, labels: np.ndarray, y_study: np.ndarray
    ) -> StudyCalibrator:
        """Fit temperature scaling on multi-label logits followed by logistic calibration on study scores.

        Args:
            logits: Array of shape (N, K) containing uncalibrated model logits.
            labels: Binary multi-label ground truth of shape (N, K).
            y_study: Binary study abnormality ground truth of shape (N,).

        Returns:
            Fitted StudyCalibrator instance.
        """
        y_labels = labels.astype(np.float64)
        y_study_arr = y_study.astype(np.float64)

        # Stage 1: Optimize scalar temperature T > 0 on multi-label BCE
        def nll_temp(params: np.ndarray) -> float:
            t = float(params[0])
            scaled_logits = logits / t
            loss = -(
                y_labels * _log_sigmoid(scaled_logits)
                + (1.0 - y_labels) * _log_one_minus_sigmoid(scaled_logits)
            )
            return float(np.mean(loss))

        res_temp = optimize.minimize(
            nll_temp,
            x0=np.array([1.0]),
            bounds=[(1e-3, 50.0)],
            method="L-BFGS-B",
        )
        self.temperature = float(res_temp.x[0])

        # Stage 2: Compute calibrated probabilities & raw study score
        calibrated_probs = _sigmoid(logits / self.temperature)
        raw_study_score = compute_study_score(calibrated_probs)
        study_logits = _logit(raw_study_score)

        # Stage 3: Fit 2-parameter logistic scaling (a * logit(p_S) + b)
        def nll_study(params: np.ndarray) -> float:
            a, b = float(params[0]), float(params[1])
            mapped_z = a * study_logits + b
            loss = -(
                y_study_arr * _log_sigmoid(mapped_z)
                + (1.0 - y_study_arr) * _log_one_minus_sigmoid(mapped_z)
            )
            return float(np.mean(loss))

        res_study = optimize.minimize(
            nll_study,
            x0=np.array([1.0, 0.0]),
            bounds=[(1e-3, 50.0), (-50.0, 50.0)],
            method="L-BFGS-B",
        )
        self.study_scale = float(res_study.x[0])
        self.study_bias = float(res_study.x[1])

        return self

    def predict_calibrated(
        self, logits: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute calibrated per-finding probabilities and study-level calibrated score.

        Args:
            logits: Uncalibrated logits array of shape (N, K).

        Returns:
            Tuple of:
                - p_findings: Calibrated multi-label probabilities in [0, 1] of shape (N, K).
                - p_study: Calibrated study abnormality probabilities in [0, 1] of shape (N,).
        """
        p_findings = _sigmoid(logits / self.temperature)
        raw_study = compute_study_score(p_findings)
        study_logits = _logit(raw_study)
        p_study = _sigmoid(self.study_scale * study_logits + self.study_bias)

        return np.clip(p_findings, 0.0, 1.0), np.clip(p_study, 0.0, 1.0)

    def save(self, path: Path) -> str:
        """Save calibrated parameters to a JSON file and return its SHA-256 hash.

        Args:
            path: Target JSON file path.

        Returns:
            SHA-256 hexadecimal string of the serialized file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "temperature": self.temperature,
            "study_scale": self.study_scale,
            "study_bias": self.study_bias,
        }
        content = json.dumps(payload, indent=2, sort_keys=True)
        path.write_text(content, encoding="utf-8")

        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: Path) -> StudyCalibrator:
        """Load calibration parameters from a JSON file.

        Args:
            path: Path to the JSON calibration file.

        Returns:
            Restored StudyCalibrator instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found at: {path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        required_keys = {"temperature", "study_scale", "study_bias"}
        missing = required_keys - set(payload.keys())
        if missing:
            raise ValueError(f"Calibration JSON missing required keys: {missing}")

        return cls(
            temperature=float(payload["temperature"]),
            study_scale=float(payload["study_scale"]),
            study_bias=float(payload["study_bias"]),
        )
