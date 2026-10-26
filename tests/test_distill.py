"""
Unit Tests for Knowledge Distillation and Feature Losses.
"""

import pytest
import torch
import torch.nn as nn
from distill.losses import HintonKDLoss, SequenceKDLoss
from distill.feature_kd import FeatureProjectionLoss, MultiLayerFeatureKDLoss
from distill.trainer import DistillationTrainer


def test_hinton_kd_soft_only():
    loss_fn = HintonKDLoss(temperature=4.0, alpha=1.0)
    student = torch.randn(4, 10, requires_grad=True)
    teacher = torch.randn(4, 10)

    loss = loss_fn(student, teacher)
    assert loss.item() >= 0.0
    loss.backward()
    assert student.grad is not None
    assert not torch.isnan(student.grad).any()


def test_hinton_kd_hard_only():
    loss_fn = HintonKDLoss(temperature=2.0, alpha=0.0)
    student = torch.randn(4, 5, requires_grad=True)
    teacher = torch.randn(4, 5)
    labels = torch.tensor([0, 1, 2, 3])

    loss = loss_fn(student, teacher, labels=labels)
    ce_expected = nn.functional.cross_entropy(student, labels)
    assert torch.isclose(loss, ce_expected, atol=1e-5)


def test_hinton_kd_temperature_scaling_invariance():
    # Identical logits with zero divergence should yield zero loss
    loss_fn = HintonKDLoss(temperature=5.0, alpha=1.0)
    logits = torch.randn(4, 8)
    loss = loss_fn(logits, logits.clone())
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_hinton_kd_shape_mismatch_raises():
    loss_fn = HintonKDLoss()
    s = torch.randn(4, 5)
    t = torch.randn(4, 8)
    with pytest.raises(ValueError, match="does not match"):
        loss_fn(s, t)


def test_hinton_kd_invalid_temperature_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        HintonKDLoss(temperature=-1.0)


def test_sequence_kd_with_attention_mask():
    loss_fn = SequenceKDLoss(temperature=2.0, alpha=1.0)
    s_logits = torch.randn(2, 6, 12, requires_grad=True)
    t_logits = torch.randn(2, 6, 12)
    # Mask out the last 2 positions
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]])

    loss = loss_fn(s_logits, t_logits, attention_mask=mask)
    assert loss.item() > 0.0
    loss.backward()
    # Check that gradients on masked positions are strictly 0.0
    assert torch.all(s_logits.grad[0, 4:, :] == 0.0)
    assert torch.all(s_logits.grad[1, 5:, :] == 0.0)


def test_feature_projection_same_dim():
    feat_loss = FeatureProjectionLoss(student_dim=32, teacher_dim=32)
    s_feat = torch.randn(4, 16, 32)
    t_feat = torch.randn(4, 16, 32)
    loss = feat_loss(s_feat, t_feat)
    assert loss.item() >= 0.0
    assert isinstance(feat_loss.projector, nn.Identity)


def test_feature_projection_diff_dim():
    feat_loss = FeatureProjectionLoss(student_dim=16, teacher_dim=64, criterion="cosine")
    s_feat = torch.randn(4, 16)
    t_feat = torch.randn(4, 64)
    loss = feat_loss(s_feat, t_feat)
    assert 0.0 <= loss.item() <= 2.0


def test_distillation_trainer_frozen_teacher():
    student = nn.Linear(10, 2)
    teacher = nn.Linear(10, 2)
    teacher_init_weight = teacher.weight.clone()

    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    trainer = DistillationTrainer(student, teacher, optimizer)

    x = torch.randn(8, 10)
    y = torch.randint(0, 2, (8,))

    metrics = trainer.train_step(x, y)
    assert metrics["step"] == 1
    assert "total_loss" in metrics

    # Verify teacher was NOT updated
    assert torch.equal(teacher.weight, teacher_init_weight)
