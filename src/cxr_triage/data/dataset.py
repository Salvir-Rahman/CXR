"""PyTorch Dataset and transformation pipeline for NIH ChestX-ray14."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms  # type: ignore[import-untyped]

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


def build_transforms(
    image_size: int = 224, is_train: bool = False
) -> Callable[[Image.Image], torch.Tensor]:
    """Build torchvision transform pipeline for training or evaluation.

    Args:
        image_size: Spatial target dimension (image_size, image_size).
        is_train: If True, applies light clinically safe augmentations
            (RandomHorizontalFlip, RandomRotation(degrees=5)).

    Returns:
        Callable torchvision composition returning a normalized FloatTensor.
    """
    transform_list: list[Any] = [
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
    ]

    if is_train:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return transforms.Compose(transform_list)


class NIHChestXrayDataset(Dataset[dict[str, Any]]):
    """PyTorch Dataset loading NIH ChestX-ray14 images and multi-label targets on demand."""

    def __init__(
        self,
        manifest_path: Path,
        images_dir: Path,
        split: Literal["train", "calibration", "test"],
        ontology_labels: tuple[str, ...],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        only_existing: bool = True,
    ) -> None:
        """Initialize the dataset from a frozen manifest.

        Args:
            manifest_path: Path to nih_splits.csv manifest.
            images_dir: Directory containing PNG images.
            split: Partition to load ('train', 'calibration', or 'test').
            ontology_labels: 14 pathology label names.
            transform: Optional image transform callable.
            only_existing: If True, filters strictly to rows where image_exists is True.
        """
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.images_dir = Path(images_dir)
        self.split = split
        self.ontology_labels = ontology_labels
        self.transform = transform
        self.only_existing = only_existing

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)
        if "split" not in df.columns:
            raise ValueError("Manifest must contain a 'split' column.")

        # Filter strictly by split
        mask = df["split"] == self.split
        if self.only_existing and "image_exists" in df.columns:
            mask = mask & df["image_exists"].astype(bool)

        self.df = df[mask].reset_index(drop=True)

        # Pre-extract numpy label matrix for fast index access
        missing_labels = set(self.ontology_labels) - set(self.df.columns)
        if missing_labels:
            raise ValueError(f"Manifest missing required ontology label columns: {missing_labels}")

        self.label_array = np.array(
            self.df[list(self.ontology_labels)].to_numpy(dtype=np.float32), copy=True
        )
        self.y_study_array = (
            self.df["y_study"].to_numpy(dtype=np.int64)
            if "y_study" in self.df.columns
            else np.zeros(len(self.df), dtype=np.int64)
        )
        self.patient_ids = self.df["Patient ID"].to_numpy(dtype=np.int64)
        self.image_ids = self.df["Image Index"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Fetch a single sample by index.

        Returns:
            Dict containing:
                - 'image': torch.FloatTensor of shape (3, H, W)
                - 'labels': torch.FloatTensor of shape (14,)
                - 'y_study': torch.LongTensor scalar (0 or 1)
                - 'patient_id': int
                - 'image_id': str
        """
        image_id = self.image_ids[idx]
        image_path = self.images_dir / image_id

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found at: {image_path}")

        with Image.open(image_path) as raw_img:
            img_rgb = raw_img.convert("RGB")

        if self.transform is not None:
            image_tensor = self.transform(img_rgb)
        else:
            image_tensor = transforms.functional.to_tensor(img_rgb)

        labels_tensor = torch.tensor(self.label_array[idx], dtype=torch.float32)
        y_study_tensor = torch.tensor(self.y_study_array[idx], dtype=torch.long)
        patient_id = int(self.patient_ids[idx])

        return {
            "image": image_tensor,
            "labels": labels_tensor,
            "y_study": y_study_tensor,
            "patient_id": patient_id,
            "image_id": image_id,
        }
