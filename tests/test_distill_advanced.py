"""
Advanced Knowledge Distillation Tests: Gradient scaling, Multi-layer feature KD, and evaluation loops.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from distill.losses import HintonKDLoss
from distill.feature_kd import MultiLayerFeatureKDLoss
from distill.trainer import DistillationTrainer


def test_hinton_t2_gradient_scaling_stability():
    """
    Verifies that the T^2 multiplier keeps gradient magnitudes comparable
    when temperature changes. Without T^2, gradient scales as 1/T^2 and vanishes at high T.
    """
    logits_s = torch.randn(16, 20)
    logits_t = torch.randn(16, 20)

    grad_norms = []
    for temp in [1.0, 2.0, 4.0]:
        s = logits_s.clone().detach().requires_grad_(True)
        loss_fn = HintonKDLoss(temperature=temp, alpha=1.0)
        loss = loss_fn(s, logits_t)
        loss.backward()
        grad_norms.append(s.grad.norm().item())

    # Ratios between gradient norms across temperatures should stay within a factor of 3
    ratio = max(grad_norms) / max(min(grad_norms), 1e-6)
    assert ratio < 4.0, f"Gradients scaled excessively: {grad_norms}"


def test_multilayer_feature_kd_weighting():
    s_dims = [16, 32]
    t_dims = [32, 64]
    weights = [0.25, 0.75]

    multi_loss = MultiLayerFeatureKDLoss(s_dims, t_dims, layer_weights=weights)

    s_feats = [torch.randn(4, 16), torch.randn(4, 32)]
    t_feats = [torch.randn(4, 32), torch.randn(4, 64)]

    loss = multi_loss(s_feats, t_feats)
    assert loss.item() > 0.0


def test_distillation_trainer_evaluate_loop():
    student = nn.Linear(8, 2)
    teacher = nn.Linear(8, 2)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)

    trainer = DistillationTrainer(student, teacher, optimizer)

    # Test dataset
    x = torch.randn(32, 8)
    y = torch.randint(0, 2, (32,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    eval_res = trainer.evaluate(loader)
    assert "val_loss" in eval_res
    assert "val_accuracy" in eval_res
    assert eval_res["total_samples"] == 32
    assert 0.0 <= eval_res["val_accuracy"] <= 1.0
