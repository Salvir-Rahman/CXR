"""Unit tests for 3-head TriageNet architecture and composite distillation loss."""

import torch

from cxr_triage.models.loss import TriageCompositeLoss
from cxr_triage.models.triage_net import TriageNet


def test_triagenet_forward_and_shapes() -> None:
    """Verify TriageNet output tensor shapes in both training and inference_only modes."""
    model = TriageNet(
        backbone_name="mobilevit_xxs",
        num_classes=14,
        pretrained=False,
        proj_dim=512,
        severity_classes=3,
    )
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    # 1. Training mode (all heads computed)
    out_train = model(x, inference_only=False)
    assert "cls_logits" in out_train
    assert "struct_logits" in out_train
    assert "proj_emb" in out_train

    assert out_train["cls_logits"].shape == (batch_size, 14)
    assert out_train["struct_logits"].shape == (batch_size, 14, 3)
    assert out_train["proj_emb"] is not None
    assert out_train["proj_emb"].shape == (batch_size, 512)

    # 2. Inference-only mode (proj_emb is None)
    out_infer = model(x, inference_only=True)
    assert out_infer["cls_logits"].shape == (batch_size, 14)
    assert out_infer["struct_logits"].shape == (batch_size, 14, 3)
    assert out_infer["proj_emb"] is None


def test_parameter_counts_compactness() -> None:
    """Verify MobileViT-XXS has < 2.5M inference parameters."""
    model = TriageNet(
        backbone_name="mobilevit_xxs",
        num_classes=14,
        pretrained=False,
        proj_dim=512,
        severity_classes=3,
    )

    counts = model.count_parameters()
    assert "total" in counts
    assert "trainable" in counts
    assert "backbone" in counts
    assert "inference_only" in counts

    assert counts["trainable"] == counts["total"]
    # Verify strict compactness constraint: inference_only < 2,500,000 parameters
    assert counts["inference_only"] < 2_500_000, (
        f"Inference parameters {counts['inference_only']} exceeds 2.5M limit!"
    )
    # Total parameter count should also be well below 2.5M (around 1.24M with projection head)
    assert counts["total"] < 2_500_000


def test_dynamic_backbone_switching() -> None:
    """Verify dynamic backbone instantiation works with alternative architectures (mobilenetv3_small_100)."""
    model = TriageNet(
        backbone_name="mobilenetv3_small_100",
        num_classes=14,
        pretrained=False,
        proj_dim=512,
        severity_classes=3,
    )

    x = torch.randn(2, 3, 224, 224)
    out = model(x, inference_only=False)

    assert out["cls_logits"].shape == (2, 14)
    assert out["struct_logits"].shape == (2, 14, 3)
    assert out["proj_emb"] is not None
    assert out["proj_emb"].shape == (2, 512)


def test_composite_loss_and_gradients() -> None:
    """Verify TriageCompositeLoss computation and backpropagation across all heads."""
    model = TriageNet(
        backbone_name="mobilevit_xxs",
        num_classes=14,
        pretrained=False,
        proj_dim=512,
        severity_classes=3,
    )
    model.train()

    loss_fn = TriageCompositeLoss(lambda_s=0.5, lambda_d=0.3)

    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 2, (batch_size, 14)).float()
    severity_targets = torch.randint(0, 3, (batch_size, 14))
    teacher_emb = torch.randn(batch_size, 512)

    # 1. Distillation ON
    out = model(x, inference_only=False)
    loss_dict = loss_fn(
        cls_logits=out["cls_logits"],
        labels=labels,
        struct_logits=out["struct_logits"],
        severity_targets=severity_targets,
        proj_emb=out["proj_emb"],
        teacher_emb=teacher_emb,
    )

    assert torch.isfinite(loss_dict["loss"])
    assert loss_dict["loss_cls"] > 0.0
    assert loss_dict["loss_struct"] > 0.0
    assert loss_dict["loss_distill"] > 0.0

    # Backpropagate and verify gradients exist
    loss_dict["loss"].backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has None grad!"
            assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite grad!"

    # 2. Distillation OFF (lambda_d=0.0)
    loss_fn_no_distill = TriageCompositeLoss(lambda_s=0.5, lambda_d=0.0)
    model.zero_grad()
    out2 = model(x, inference_only=False)
    loss_dict2 = loss_fn_no_distill(
        cls_logits=out2["cls_logits"],
        labels=labels,
        struct_logits=out2["struct_logits"],
        severity_targets=severity_targets,
        proj_emb=out2["proj_emb"],
        teacher_emb=teacher_emb,
    )
    assert loss_dict2["loss_distill"].item() == 0.0
    assert torch.isfinite(loss_dict2["loss"])

    # 3. Structural loss OFF (severity_targets=None)
    loss_dict3 = loss_fn_no_distill(
        cls_logits=out2["cls_logits"],
        labels=labels,
        struct_logits=None,
        severity_targets=None,
    )
    assert loss_dict3["loss_struct"].item() == 0.0
    assert loss_dict3["loss_distill"].item() == 0.0
    assert torch.isclose(loss_dict3["loss"], loss_dict3["loss_cls"])
