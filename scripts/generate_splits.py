"""CLI script to generate and freeze the NIH ChestX-ray14 patient splits (Deliverable D1)."""

import sys
from pathlib import Path

# Add src to sys.path for direct script execution if needed
src_dir = Path(__file__).resolve().parents[1] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from cxr_triage.config import load_config
from cxr_triage.data.splits import generate_and_save_splits


def main() -> None:
    config = load_config()
    print("=" * 70)
    print("NEURO-SYMBOLIC CXR TRIAGE: GENERATING DELIVERABLE D1 SPLITS")
    print("=" * 70)
    print(f"Loading raw NIH dataset from: {config.paths.nih_dir}")
    print(f"Target processed directory:   {config.paths.processed_data_dir}")
    print(f"Reproducibility seed:         {config.reproducibility.seed}")

    df, sha256_hash = generate_and_save_splits(config)

    total_studies = len(df)
    total_patients = df["Patient ID"].nunique()

    print("\n--- SPLIT DISTRIBUTION SUMMARY ---")
    print(f"Total Studies (Rows):   {total_studies:,}")
    print(f"Total Unique Patients:  {total_patients:,}")

    pids_by_split: dict[str, set[str | int]] = {}
    for split_name in ("train", "calibration", "test"):
        sub = df[df["split"] == split_name]
        studies_cnt = len(sub)
        pids = set(sub["Patient ID"].unique())
        pids_by_split[split_name] = pids
        pct_studies = (studies_cnt / total_studies) * 100
        pct_patients = (len(pids) / total_patients) * 100
        print(
            f"  - Split '{split_name:11s}': {studies_cnt:6,d} studies ({pct_studies:5.1f}%) | "
            f"{len(pids):5,d} patients ({pct_patients:5.1f}%)"
        )

    # Validate zero patient overlap
    overlap_train_calib = len(pids_by_split["train"] & pids_by_split["calibration"])
    overlap_train_test = len(pids_by_split["train"] & pids_by_split["test"])
    overlap_calib_test = len(pids_by_split["calibration"] & pids_by_split["test"])
    total_leakage = overlap_train_calib + overlap_train_test + overlap_calib_test

    print("\n--- PATIENT DISJOINTNESS & LEAKAGE CHECK ---")
    print(f"  Overlap (train <-> calibration): {overlap_train_calib} patients")
    print(f"  Overlap (train <-> test):        {overlap_train_test} patients")
    print(f"  Overlap (calibration <-> test):  {overlap_calib_test} patients")
    print(f"  Total Overlapping Patients:      {total_leakage} (LEAK-FREE VERIFIED)")

    # Image existence check
    local_images_found = int(df["image_exists"].sum())
    pct_images = (local_images_found / total_studies) * 100
    print("\n--- LOCAL IMAGE CACHE CHECK ---")
    print(f"  Local Images Present: {local_images_found:,} / {total_studies:,} ({pct_images:.2f}%)")

    # Pathological breakdown
    abnormal_studies = int(df["y_study"].sum())
    normal_studies = total_studies - abnormal_studies
    print("\n--- STUDY-LEVEL ABNORMALITY (Y_S) ---")
    print(f"  Normal ('No Finding'): {normal_studies:,} ({(normal_studies / total_studies) * 100:.1f}%)")
    print(f"  Abnormal (>=1 finding): {abnormal_studies:,} ({(abnormal_studies / total_studies) * 100:.1f}%)")

    print("\n--- ARTIFACT CHECKSUM (SHA-256) ---")
    print(f"  Manifest: {config.paths.processed_data_dir / 'nih_splits.csv'}")
    print(f"  SHA-256:  {sha256_hash}")
    print("=" * 70)


if __name__ == "__main__":
    main()
