"""Unit and integration tests for patient-level split generator (Deliverable D1)."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from cxr_triage.config import SplitsConfig, load_config
from cxr_triage.data.splits import (
    check_image_existence,
    compute_file_sha256,
    create_patient_splits,
    generate_and_save_splits,
    parse_finding_labels,
)


def test_parse_finding_labels_edge_cases() -> None:
    """Test parsing of 'No Finding', single findings, and multiple combined findings."""
    ontology_labels = (
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
    )

    series = pd.Series([
        "No Finding",
        "Cardiomegaly",
        "Cardiomegaly|Effusion",
        "Pneumothorax|Edema|Pneumonia",
        "   ",
        None,
    ])

    parsed = parse_finding_labels(series, ontology_labels)

    # 1. 'No Finding' -> all zeros
    assert parsed.loc[0, list(ontology_labels)].sum() == 0
    assert parsed.loc[0, "y_study"] == 0
    assert parsed.loc[0, "finding_count"] == 0

    # 2. 'Cardiomegaly' -> 1 finding
    assert parsed.loc[1, "Cardiomegaly"] == 1
    assert parsed.loc[1, "Effusion"] == 0
    assert parsed.loc[1, "y_study"] == 1
    assert parsed.loc[1, "finding_count"] == 1

    # 3. 'Cardiomegaly|Effusion' -> 2 findings
    assert parsed.loc[2, "Cardiomegaly"] == 1
    assert parsed.loc[2, "Effusion"] == 1
    assert parsed.loc[2, "Atelectasis"] == 0
    assert parsed.loc[2, "y_study"] == 1
    assert parsed.loc[2, "finding_count"] == 2

    # 4. 'Pneumothorax|Edema|Pneumonia' -> 3 findings
    assert parsed.loc[3, "Pneumothorax"] == 1
    assert parsed.loc[3, "Edema"] == 1
    assert parsed.loc[3, "Pneumonia"] == 1
    assert parsed.loc[3, "y_study"] == 1
    assert parsed.loc[3, "finding_count"] == 3

    # 5. Empty / None -> zero findings
    assert parsed.loc[4, "y_study"] == 0
    assert parsed.loc[5, "y_study"] == 0


def test_patient_level_split_disjointness() -> None:
    """Verify that multiple studies from the same patient always land in the same split."""
    # Synthetic dataset with 100 patients, each having between 1 and 5 studies
    records = []
    for pid in range(1, 101):
        num_studies = (pid % 5) + 1
        for sid in range(num_studies):
            records.append({
                "Image Index": f"{pid:08d}_{sid:03d}.png",
                "Patient ID": pid,
            })
    df = pd.DataFrame(records)

    splits_cfg = SplitsConfig(train=0.70, val=0.15, test=0.15)
    split_df = create_patient_splits(df, splits_cfg, seed=42)

    train_pids = set(split_df[split_df["split"] == "train"]["Patient ID"])
    calib_pids = set(split_df[split_df["split"] == "calibration"]["Patient ID"])
    test_pids = set(split_df[split_df["split"] == "test"]["Patient ID"])

    # Disjointness checks
    assert train_pids.isdisjoint(calib_pids), "Leakage between train and calibration!"
    assert train_pids.isdisjoint(test_pids), "Leakage between train and test!"
    assert calib_pids.isdisjoint(test_pids), "Leakage between calibration and test!"

    # Completeness check
    assert len(train_pids) + len(calib_pids) + len(test_pids) == 100
    assert len(split_df) == len(df)
    assert set(split_df["split"].unique()) == {"train", "calibration", "test"}


def test_split_determinism() -> None:
    """Verify that the same seed produces bit-for-bit identical patient assignments."""
    df = pd.DataFrame({
        "Patient ID": np.repeat(np.arange(50), 3),
        "Image Index": [f"img_{i}.png" for i in range(150)],
    })

    splits_cfg = SplitsConfig(train=0.70, val=0.15, test=0.15)
    split_1 = create_patient_splits(df, splits_cfg, seed=123)
    split_2 = create_patient_splits(df, splits_cfg, seed=123)
    split_3 = create_patient_splits(df, splits_cfg, seed=999)

    pd.testing.assert_series_equal(split_1["split"], split_2["split"])
    assert not (split_1["split"] == split_3["split"]).all()


def test_check_image_existence(tmp_path: Path) -> None:
    """Verify local image existence checking."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "00000001_000.png").touch()
    (img_dir / "00000002_000.png").touch()

    series = pd.Series(["00000001_000.png", "00000002_000.png", "00000003_000.png"])
    exists = check_image_existence(series, img_dir)

    assert list(exists) == [True, True, False]


def test_compute_file_sha256(tmp_path: Path) -> None:
    """Verify sha256 calculation matches hashlib on known content."""
    dummy_file = tmp_path / "test.txt"
    content = b"Neuro-Symbolic CXR Triage"
    dummy_file.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    assert compute_file_sha256(dummy_file) == expected


def test_real_nih_manifest_generation() -> None:
    """Integration test against actual Data_Entry_2017.csv if present."""
    config = load_config()
    raw_csv = config.paths.nih_dir / "Data_Entry_2017.csv"
    if not raw_csv.exists():
        pytest.skip(f"Raw NIH data entry file not found at {raw_csv}")

    df, sha256_hash = generate_and_save_splits(config)

    assert len(df) == 112120
    assert df["Patient ID"].nunique() == 30805

    # 14 binary labels present
    for label in config.ontology.labels:
        assert label in df.columns
        assert set(df[label].unique()).issubset({0, 1})

    assert "y_study" in df.columns
    assert "finding_count" in df.columns
    assert "split" in df.columns
    assert "image_exists" in df.columns

    # 0 patient overlap
    pids_train = set(df[df["split"] == "train"]["Patient ID"])
    pids_calib = set(df[df["split"] == "calibration"]["Patient ID"])
    pids_test = set(df[df["split"] == "test"]["Patient ID"])

    assert pids_train.isdisjoint(pids_calib)
    assert pids_train.isdisjoint(pids_test)
    assert pids_calib.isdisjoint(pids_test)

    # 4999 local images exist
    assert int(df["image_exists"].sum()) == 4999

    # SHA256 checksum file exists and matches
    sha_file = config.paths.processed_data_dir / "nih_splits.sha256"
    assert sha_file.exists()
    assert sha256_hash in sha_file.read_text(encoding="utf-8")
