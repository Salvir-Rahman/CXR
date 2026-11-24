"""Compact 3-head hybrid vision architecture (MobileViT-XXS) for chest X-ray triage."""

from __future__ import annotations

import timm  # type: ignore[import-untyped]
import torch
from torch import nn


class TriageNet(nn.Module):
    """3-Head compact neural perception backbone with feature distillation support.

    Heads:
        - `cls_head`: Multi-label binary disease classification logits (B, num_classes).
        - `struct_head`: Discretized 3-state pathology severity logits (B, num_classes, 3).
        - `proj_head`: 2-layer projection MLP for multimodal teacher embedding alignment (B, proj_dim).
    """

    def __init__(
        self,
        backbone_name: str = "mobilevit_xxs",
        num_classes: int = 14,
        pretrained: bool = False,
        proj_dim: int = 512,
        severity_classes: int = 3,
    ) -> None:
        """Initialize TriageNet with dynamic backbone and 3 task heads.

        Args:
            backbone_name: Name of timm vision model (e.g. 'mobilevit_xxs', 'mobilenetv3_small_100').
            num_classes: Number of pathology classes (NIH-14 = 14).
            pretrained: Whether to load ImageNet pretrained weights.
            proj_dim: Dimension of multimodal teacher projection embedding (BiomedCLIP = 512).
            severity_classes: Number of discrete severity states (3 = absent, present, severe).
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.severity_classes = severity_classes

        # Dynamically instantiate backbone with pooled feature vector output
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )

        # Robust dynamic feature dimension determination across all timm model families
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            self.in_features: int = int(self.backbone(dummy).shape[-1])

        # 1. Multi-label pathology head
        self.cls_head = nn.Linear(self.in_features, num_classes)

        # 2. Structured 3-state severity head
        self.struct_head = nn.Linear(self.in_features, num_classes * severity_classes)

        # 3. Multimodal alignment projection head (2-layer MLP with GELU)
        self.proj_head = nn.Sequential(
            nn.Linear(self.in_features, self.in_features),
            nn.GELU(),
            nn.Linear(self.in_features, proj_dim),
        )

    def forward(
        self, x: torch.Tensor, inference_only: bool = False
    ) -> dict[str, torch.Tensor | None]:
        """Forward pass through backbone and task heads.

        Args:
            x: Input tensor of shape (B, 3, H, W).
            inference_only: If True, skips projection head computation (saves latency/memory).

        Returns:
            Dict containing:
                - 'cls_logits': FloatTensor of shape (B, num_classes)
                - 'struct_logits': FloatTensor of shape (B, num_classes, severity_classes)
                - 'proj_emb': FloatTensor of shape (B, proj_dim) if not inference_only else None
        """
        features: torch.Tensor = self.backbone(x)

        cls_logits = self.cls_head(features)
        struct_raw = self.struct_head(features)
        struct_logits = struct_raw.view(
            -1, self.num_classes, self.severity_classes
        )

        proj_emb = None if inference_only else self.proj_head(features)

        return {
            "cls_logits": cls_logits,
            "struct_logits": struct_logits,
            "proj_emb": proj_emb,
        }

    def count_parameters(self) -> dict[str, int]:
        """Compute exact parameter statistics across submodules."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_count = sum(p.numel() for p in self.backbone.parameters())
        cls_count = sum(p.numel() for p in self.cls_head.parameters())
        struct_count = sum(p.numel() for p in self.struct_head.parameters())
        inference_only = backbone_count + cls_count + struct_count

        return {
            "total": total,
            "trainable": trainable,
            "backbone": backbone_count,
            "inference_only": inference_only,
        }
