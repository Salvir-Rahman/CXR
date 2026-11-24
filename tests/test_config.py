"""Unit tests for the typed, immutable configuration loader."""

import dataclasses
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from cxr_triage.config import (
    AppConfig,
    load_config,
    validate_config,
)


def test_load_default_config() -> None:
    """Verify loading the actual default.yaml matches the revised proposal."""
    config = load_config()

    assert isinstance(config, AppConfig)
    assert config.project.name == "neuro-symbolic-cxr-triage"
    assert config.model.backbone == "mobilevit_xxs"
    assert config.model.in_channels == 3
    assert config.model.num_classes == 14
    assert config.data.splits.train == 0.70
    assert config.data.splits.val == 0.15
    assert config.data.splits.test == 0.15
    assert math.isclose(
        config.data.splits.train + config.data.splits.val + config.data.splits.test,
        1.0,
        abs_tol=1e-5,
    )
    assert config.distillation.teacher == "BiomedCLIP"
    assert config.distillation.lambda_s == 0.5
    assert config.distillation.lambda_d == 0.3
    assert config.study_parameters.epsilon == 0.02
    assert config.study_parameters.delta == 0.05
    assert config.study_parameters.c_min == 0.30
    assert len(config.ontology.labels) == 14
    assert "Pneumothorax" in config.ontology.critical_findings
    assert "Edema" in config.ontology.critical_findings


def test_config_immutability() -> None:
    """Verify that dataclass instances are strictly frozen/immutable."""
    config = load_config()

    with pytest.raises(FrozenInstanceError):
        # type: ignore[misc]
        config.model.backbone = "resnet50"  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        # type: ignore[misc]
        config.data.splits.train = 0.80  # type: ignore[misc]

    # Verify tuples cannot be mutated in place
    assert isinstance(config.ontology.labels, tuple)
    assert isinstance(config.ontology.critical_findings, tuple)
    assert isinstance(config.study_parameters.coverage_grid, tuple)


def test_dynamic_path_resolution() -> None:
    """Verify paths are resolved dynamically without hardcoded absolute paths."""
    config = load_config()

    assert config.paths.root_dir.is_absolute()
    assert config.paths.nih_dir == config.paths.root_dir / "data" / "raw" / "nih"
    assert config.paths.raw_data_dir == config.paths.root_dir / "data" / "raw"
    assert config.paths.artifacts_dir == config.paths.root_dir / "artifacts"


def test_custom_project_root(tmp_path: Path) -> None:
    """Verify load_config honors custom config path and dynamic root resolution."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    custom_yaml = config_dir / "custom.yaml"

    # Read base default.yaml and write to custom location
    base_cfg = load_config()
    with open(base_cfg.paths.root_dir / "configs" / "default.yaml", encoding="utf-8") as f:
        content = f.read()
    custom_yaml.write_text(content, encoding="utf-8")

    loaded = load_config(config_path=custom_yaml, project_root=tmp_path)
    assert loaded.paths.root_dir == tmp_path.resolve()
    assert loaded.paths.nih_dir == tmp_path.resolve() / "data" / "raw" / "nih"


def test_splits_validation_sum() -> None:
    """Verify invalid split ratios summing != 1.0 raise ValueError."""
    config = load_config()
    invalid_splits = dataclasses.replace(config.data.splits, train=0.60)  # sums to 0.90
    invalid_data = dataclasses.replace(config.data, splits=invalid_splits)
    invalid_config = dataclasses.replace(config, data=invalid_data)

    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        validate_config(invalid_config)


def test_splits_validation_positive() -> None:
    """Verify non-positive split ratios raise ValueError."""
    config = load_config()
    invalid_splits = dataclasses.replace(config.data.splits, train=-0.1, val=0.55, test=0.55)
    invalid_data = dataclasses.replace(config.data, splits=invalid_splits)
    invalid_config = dataclasses.replace(config, data=invalid_data)

    with pytest.raises(ValueError, match="All split ratios must be positive"):
        validate_config(invalid_config)


def test_study_parameters_bounds() -> None:
    """Verify epsilon, delta, and c_min bounds are strictly enforced."""
    config = load_config()

    # epsilon <= 0
    with pytest.raises(ValueError, match="epsilon must be strictly between 0 and 1"):
        validate_config(dataclasses.replace(
            config,
            study_parameters=dataclasses.replace(config.study_parameters, epsilon=0.0)
        ))

    # epsilon >= 1
    with pytest.raises(ValueError, match="epsilon must be strictly between 0 and 1"):
        validate_config(dataclasses.replace(
            config,
            study_parameters=dataclasses.replace(config.study_parameters, epsilon=1.0)
        ))

    # delta <= 0
    with pytest.raises(ValueError, match="delta must be strictly between 0 and 1"):
        validate_config(dataclasses.replace(
            config,
            study_parameters=dataclasses.replace(config.study_parameters, delta=-0.05)
        ))

    # c_min <= 0
    with pytest.raises(ValueError, match="c_min must be in"):
        validate_config(dataclasses.replace(
            config,
            study_parameters=dataclasses.replace(config.study_parameters, c_min=0.0)
        ))


def test_ontology_labels_validation() -> None:
    """Verify label count and naming integrity."""
    config = load_config()

    # Missing label (count 13)
    short_labels = config.ontology.labels[:-1]
    with pytest.raises(ValueError, match="Expected 14 labels"):
        validate_config(dataclasses.replace(
            config,
            ontology=dataclasses.replace(config.ontology, labels=short_labels)
        ))

    # Wrong label name
    mutated_labels = ("UnknownDisease",) + config.ontology.labels[1:]
    with pytest.raises(ValueError, match="Labels do not match standard NIH-ChestXray14"):
        validate_config(dataclasses.replace(
            config,
            ontology=dataclasses.replace(config.ontology, labels=mutated_labels)
        ))


def test_critical_findings_validation() -> None:
    """Verify critical findings must be a subset of ontology labels."""
    config = load_config()

    with pytest.raises(ValueError, match="critical_findings must be a subset of ontology labels"):
        validate_config(dataclasses.replace(
            config,
            ontology=dataclasses.replace(
                config.ontology,
                critical_findings=("UnknownCriticalFinding",)
            )
        ))


def test_model_validation() -> None:
    """Verify model channel and num_classes checks."""
    config = load_config()

    with pytest.raises(ValueError, match="in_channels must be 1 or 3"):
        validate_config(dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, in_channels=4)
        ))

    with pytest.raises(ValueError, match="num_classes.*must match label count"):
        validate_config(dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, num_classes=10)
        ))


def test_distillation_validation() -> None:
    """Verify distillation parameters."""
    config = load_config()

    with pytest.raises(ValueError, match="lambda_s must be non-negative"):
        validate_config(dataclasses.replace(
            config,
            distillation=dataclasses.replace(config.distillation, lambda_s=-0.1)
        ))

    with pytest.raises(ValueError, match="lambda_d must be non-negative"):
        validate_config(dataclasses.replace(
            config,
            distillation=dataclasses.replace(config.distillation, lambda_d=-0.1)
        ))

    with pytest.raises(ValueError, match="teacher must be a non-empty string"):
        validate_config(dataclasses.replace(
            config,
            distillation=dataclasses.replace(config.distillation, teacher="   ")
        ))
