"""Leak-free patient-level split generator and NIH ChestX-ray14 label parser."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from cxr_triage.config import AppConfig, SplitsConfig, load_config


def parse_finding_labels(
    labels_series: pd.Series, ontology_labels: tuple[str, ...]
) -> pd.DataFrame:
    r"""Parse pipe-delimited finding label strings into binary multi-label vectors.

    Produces:
        - 14 binary columns matching `ontology_labels` (0 or 1)
        - `finding_count`: total count of positive pathology labels ($N = \sum Y_k$)
        - `y_study`: study-level binary abnormality indicator ($Y_S = \mathbb{I}[N > 0]$)

    Args:
        labels_series: Series containing NIH Finding Labels strings (e.g. "Cardiomegaly|Effusion").
        ontology_labels: Canonical list of the 14 NIH pathology labels.

    Returns:
        DataFrame with the 14 binary label columns, `finding_count`, and `y_study`.
    """
    tokenized = (
        labels_series.fillna("No Finding")
        .astype(str)
        .str.split("|")
        .apply(lambda items: {x.strip() for x in items})
    )

    data: dict[str, pd.Series] = {}
    for label in ontology_labels:
        data[label] = tokenized.apply(lambda s, lbl=label: 1 if lbl in s else 0).astype(  # type: ignore[misc]
            "int64"
        )

    labels_df = pd.DataFrame(data, index=labels_series.index)
    finding_count = labels_df[list(ontology_labels)].sum(axis=1).astype("int64")
    labels_df["finding_count"] = finding_count
    labels_df["y_study"] = (finding_count > 0).astype("int64")

    return labels_df


def create_patient_splits(
    df: pd.DataFrame, splits_cfg: SplitsConfig, seed: int
) -> pd.DataFrame:
    """Partition dataset strictly by Patient ID to prevent patient-level data leakage.

    Splits are allocated as:
        - `train`: `splits_cfg.train` (e.g. 70%)
        - `calibration`: `splits_cfg.val` (e.g. 15%)
        - `test`: `splits_cfg.test` (e.g. 15%)

    Args:
        df: DataFrame containing at least a 'Patient ID' column.
        splits_cfg: SplitsConfig specifying train, val (calibration), and test fractions.
        seed: Random seed for deterministic reproducibility.

    Returns:
        DataFrame with a new 'split' column containing values 'train', 'calibration', or 'test'.
    """
    if "Patient ID" not in df.columns:
        raise KeyError("Dataframe must contain 'Patient ID' column for patient-level partitioning.")

    # Sort unique patient IDs to guarantee platform-independent determinism before shuffle
    unique_patients = np.sort(df["Patient ID"].unique())
    n_patients = len(unique_patients)

    rng = np.random.default_rng(seed)
    shuffled_patients = rng.permutation(unique_patients)

    n_train = int(n_patients * splits_cfg.train)
    n_calib = int(n_patients * splits_cfg.val)

    train_pids = set(shuffled_patients[:n_train])
    calib_pids = set(shuffled_patients[n_train : n_train + n_calib])
    test_pids = set(shuffled_patients[n_train + n_calib :])

    # Mathematical disjointness assertion
    assert not (train_pids & calib_pids), "Patient leakage between train and calibration!"
    assert not (train_pids & test_pids), "Patient leakage between train and test!"
    assert not (calib_pids & test_pids), "Patient leakage between calibration and test!"
    assert len(train_pids) + len(calib_pids) + len(test_pids) == n_patients

    split_map: dict[int | str, str] = {}
    for pid in train_pids:
        split_map[pid] = "train"
    for pid in calib_pids:
        split_map[pid] = "calibration"
    for pid in test_pids:
        split_map[pid] = "test"

    out_df = df.copy()
    out_df["split"] = out_df["Patient ID"].map(split_map)
    return out_df


def check_image_existence(image_index_series: pd.Series, images_dir: Path) -> pd.Series:
    """Check which image files exist on the local disk.

    Args:
        image_index_series: Series containing image filenames (e.g. "00000001_000.png").
        images_dir: Directory where uncompressed image files reside.

    Returns:
        Boolean Series indicating existence for each row.
    """
    if not images_dir.exists():
        return pd.Series(False, index=image_index_series.index, dtype=bool)

    # Fast hash-set lookup
    existing_filenames = set(os.listdir(images_dir))
    return image_index_series.isin(existing_filenames)


def compute_file_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hexadecimal digest of a file in 64KB chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_and_save_splits(
    config: AppConfig | None = None,
) -> tuple[pd.DataFrame, str]:
    """Generate the full frozen patient split manifest and write to disk with SHA-256.

    Args:
        config: Application configuration. If None, loaded via `load_config()`.

    Returns:
        Tuple of (processed DataFrame, sha256_hash string).
    """
    if config is None:
        config = load_config()

    csv_path = config.paths.nih_dir / "Data_Entry_2017.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw NIH data entry file not found at: {csv_path}")

    df_raw = pd.read_csv(csv_path)

    # 1. Multi-label parsing
    parsed_labels = parse_finding_labels(
        df_raw["Finding Labels"], config.ontology.labels
    )

    # 2. Patient-level split assignment
    df_merged = pd.concat([df_raw, parsed_labels], axis=1)
    df_splits = create_patient_splits(
        df_merged, config.data.splits, config.reproducibility.seed
    )

    # 3. Dynamic image existence check
    images_dir = config.paths.nih_dir / "images"
    df_splits["image_exists"] = check_image_existence(
        df_splits["Image Index"], images_dir
    )

    # 4. Save to processed data directory
    out_dir = config.paths.processed_data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "nih_splits.csv"
    df_splits.to_csv(manifest_path, index=False)

    sha256_hash = compute_file_sha256(manifest_path)
    sha256_path = out_dir / "nih_splits.sha256"
    sha256_path.write_text(f"{sha256_hash}  nih_splits.csv\n", encoding="utf-8")

    return df_splits, sha256_hash
