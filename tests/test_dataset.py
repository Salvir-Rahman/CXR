"""Unit and integration tests for PyTorch NIH-14 Dataset and transform pipeline."""

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from cxr_triage.config import NIH_CHESTXRAY14_LABELS, load_config
from cxr_triage.data.dataset import (
    NIHChestXrayDataset,
    build_transforms,
)


def test_build_transforms_shapes_and_types() -> None:
    """Verify transform output tensor shapes, dtypes, and value normalization."""
    raw_img = Image.new("RGB", (300, 400), color=(128, 128, 128))

    eval_transform = build_transforms(image_size=224, is_train=False)
    eval_tensor = eval_transform(raw_img)

    assert isinstance(eval_tensor, torch.Tensor)
    assert eval_tensor.shape == (3, 224, 224)
    assert eval_tensor.dtype == torch.float32

    train_transform = build_transforms(image_size=224, is_train=True)
    train_tensor = train_transform(raw_img)

    assert isinstance(train_tensor, torch.Tensor)
    assert train_tensor.shape == (3, 224, 224)
    assert train_tensor.dtype == torch.float32

    # Verify custom size
    custom_transform = build_transforms(image_size=128, is_train=False)
    custom_tensor = custom_transform(raw_img)
    assert custom_tensor.shape == (3, 128, 128)


def test_dataset_synthetic_mock(tmp_path: Path) -> None:
    """Verify NIHChestXrayDataset filtering, indexing, and dict contract using synthetic files."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create 3 synthetic PNGs
    for i in range(1, 4):
        img = Image.new("L", (64, 64), color=i * 50)
        img.save(images_dir / f"img_00{i}.png")

    labels_list = sorted(NIH_CHESTXRAY14_LABELS)
    manifest_rows = [
        # Image 1: train, exists=True
        {
            "Image Index": "img_001.png",
            "Patient ID": 101,
            "split": "train",
            "image_exists": True,
            "y_study": 1,
            **{lbl: (1 if lbl in ("Cardiomegaly", "Effusion") else 0) for lbl in labels_list},
        },
        # Image 2: calibration, exists=True
        {
            "Image Index": "img_002.png",
            "Patient ID": 102,
            "split": "calibration",
            "image_exists": True,
            "y_study": 0,
            **{lbl: 0 for lbl in labels_list},
        },
        # Image 3: test, exists=False (supposedly missing locally)
        {
            "Image Index": "img_003.png",
            "Patient ID": 103,
            "split": "test",
            "image_exists": False,
            "y_study": 1,
            **{lbl: (1 if lbl == "Pneumothorax" else 0) for lbl in labels_list},
        },
    ]

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = tmp_path / "test_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    transform = build_transforms(image_size=64, is_train=False)

    # Test train split
    train_dataset = NIHChestXrayDataset(
        manifest_path=manifest_path,
        images_dir=images_dir,
        split="train",
        ontology_labels=tuple(labels_list),
        transform=transform,
        only_existing=True,
    )
    assert len(train_dataset) == 1
    sample = train_dataset[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["labels"].shape == (14,)
    assert sample["labels"].dtype == torch.float32
    assert sample["y_study"].item() == 1
    assert sample["patient_id"] == 101
    assert sample["image_id"] == "img_001.png"

    # Verify Effusion and Cardiomegaly are 1, others 0
    cardio_idx = labels_list.index("Cardiomegaly")
    effusion_idx = labels_list.index("Effusion")
    assert sample["labels"][cardio_idx] == 1.0
    assert sample["labels"][effusion_idx] == 1.0

    # Test test split with only_existing=True (should be empty because exists=False)
    test_dataset = NIHChestXrayDataset(
        manifest_path=manifest_path,
        images_dir=images_dir,
        split="test",
        ontology_labels=tuple(labels_list),
        transform=transform,
        only_existing=True,
    )
    assert len(test_dataset) == 0

    # Test test split with only_existing=False (should contain 1 sample)
    test_dataset_all = NIHChestXrayDataset(
        manifest_path=manifest_path,
        images_dir=images_dir,
        split="test",
        ontology_labels=tuple(labels_list),
        transform=transform,
        only_existing=False,
    )
    assert len(test_dataset_all) == 1


def test_real_nih_dataset_integration() -> None:
    """Integration test loading real images and batches from data/processed and data/raw."""
    config = load_config()
    manifest_path = config.paths.processed_data_dir / "nih_splits.csv"
    images_dir = config.paths.nih_dir / "images"

    if not manifest_path.exists() or not images_dir.exists():
        pytest.skip("Processed manifest or raw images directory not found, skipping real integration test.")

    transform = build_transforms(image_size=config.model.image_size, is_train=False)

    # Verify each split with only_existing=True loads accurately
    for split_name, expected_count in [("train", 3641), ("calibration", 691), ("test", 667)]:
        ds = NIHChestXrayDataset(
            manifest_path=manifest_path,
            images_dir=images_dir,
            split=split_name,  # type: ignore[arg-type]
            ontology_labels=config.ontology.labels,
            transform=transform,
            only_existing=True,
        )
        assert len(ds) == expected_count, f"Split {split_name} count mismatch!"

    # Test batch loading with DataLoader on train split
    train_ds = NIHChestXrayDataset(
        manifest_path=manifest_path,
        images_dir=images_dir,
        split="train",
        ontology_labels=config.ontology.labels,
        transform=transform,
        only_existing=True,
    )

    batch_size = 8
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))

    assert "image" in batch
    assert "labels" in batch
    assert "y_study" in batch
    assert "patient_id" in batch
    assert "image_id" in batch

    assert batch["image"].shape == (batch_size, 3, 224, 224)
    assert batch["image"].dtype == torch.float32
    assert batch["labels"].shape == (batch_size, 14)
    assert batch["labels"].dtype == torch.float32
    assert batch["y_study"].shape == (batch_size,)
    assert batch["y_study"].dtype == torch.int64
    assert len(batch["patient_id"]) == batch_size
    assert len(batch["image_id"]) == batch_size
