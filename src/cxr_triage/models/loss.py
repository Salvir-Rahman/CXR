"""Composite loss module integrating multi-label classification, structural severity, and distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TriageCompositeLoss(nn.Module):
    r"""Composite multi-task and distillation loss for Neuro-Symbolic CXR Triage.

    Loss Formulation:
        \mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda_s \mathcal{L}_{\text{struct}} + \lambda_d \mathcal{L}_{\text{distill}}

    Components:
        - \mathcal{L}_{\text{cls}}: BCEWithLogitsLoss over binary multi-label targets (B, 14).
        - \mathcal{L}_{\text{struct}}: CrossEntropyLoss over 3-state severity distributions (B, 14, 3).
        - \mathcal{L}_{\text{distill}}: MSELoss aligning student projection with multimodal teacher embeddings (B, 512).
    """

    def __init__(self, lambda_s: float = 0.5, lambda_d: float = 0.3) -> None:
        """Initialize loss weights.

        Args:
            lambda_s: Weight for structural 3-state severity classification loss.
            lambda_d: Weight for teacher feature distillation alignment loss.
        """
        super().__init__()
        self.lambda_s = float(lambda_s)
        self.lambda_d = float(lambda_d)

    def forward(
        self,
        cls_logits: torch.Tensor,
        labels: torch.Tensor,
        struct_logits: torch.Tensor | None = None,
        severity_targets: torch.Tensor | None = None,
        proj_emb: torch.Tensor | None = None,
        teacher_emb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute composite loss and individual loss components.

        Args:
            cls_logits: Binary classification logits of shape (B, num_classes).
            labels: Binary multi-label ground truth of shape (B, num_classes).
            struct_logits: Optional severity logits of shape (B, num_classes, 3).
            severity_targets: Optional discrete severity targets of shape (B, num_classes) in {0, 1, 2}.
            proj_emb: Optional student projection embeddings of shape (B, proj_dim).
            teacher_emb: Optional teacher target embeddings of shape (B, proj_dim).

        Returns:
            Dict containing:
                - 'loss': Total weighted composite loss scalar.
                - 'loss_cls': Classification BCE loss scalar.
                - 'loss_struct': Structural severity CE loss scalar.
                - 'loss_distill': Distillation MSE loss scalar.
        """
        device = cls_logits.device
        dtype = cls_logits.dtype

        # 1. Primary multi-label binary cross-entropy loss
        loss_cls = F.binary_cross_entropy_with_logits(cls_logits, labels.float())

        # 2. Secondary structural 3-state severity loss
        if (
            struct_logits is not None
            and severity_targets is not None
            and self.lambda_s > 0.0
        ):
            num_severity_classes = struct_logits.shape[-1]
            loss_struct = F.cross_entropy(
                struct_logits.reshape(-1, num_severity_classes),
                severity_targets.reshape(-1).long(),
            )
        else:
            loss_struct = torch.tensor(0.0, device=device, dtype=dtype)

        # 3. Tertiary multimodal feature distillation loss
        if (
            proj_emb is not None
            and teacher_emb is not None
            and self.lambda_d > 0.0
        ):
            loss_distill = F.mse_loss(proj_emb, teacher_emb)
        else:
            loss_distill = torch.tensor(0.0, device=device, dtype=dtype)

        # Composite total
        total_loss = (
            loss_cls
            + (self.lambda_s * loss_struct)
            + (self.lambda_d * loss_distill)
        )

        return {
            "loss": total_loss,
            "loss_cls": loss_cls,
            "loss_struct": loss_struct,
            "loss_distill": loss_distill,
        }
