"""
Comprehensive Edge Case and Robustness Tests for PTQ Quantization.
"""

import pytest
import torch
from quantize.numerics import (
    quantize_symmetric,
    dequantize_symmetric,
    quantize_asymmetric,
    dequantize_asymmetric,
)
from quantize.granularity import (
    quantize_per_tensor,
    quantize_per_channel,
    quantize_grouped,
    dequantize_grouped,
)
from quantize.calibration import calibrate_percentile, calibrate_mse


def test_quantize_non_contiguous_tensor():
    # Create transposed tensor with non-contiguous memory stride
    x = torch.randn(10, 20).t()
    assert not x.is_contiguous()

    q, scale = quantize_symmetric(x, bits=8)
    recon = dequantize_symmetric(q, scale)
    assert recon.shape == x.shape
    assert torch.allclose(x, recon, atol=0.05)


def test_quantize_multidimensional_4d_conv():
    # 4D Conv weight: (out_channels, in_channels, kH, kW)
    weights = torch.randn(16, 32, 3, 3)
    q, scale, _ = quantize_per_channel(weights, dim=0, bits=8, symmetric=True)

    assert scale.shape == (16, 1, 1, 1)
    recon = dequantize_symmetric(q, scale)
    assert recon.shape == weights.shape
    assert torch.allclose(weights, recon, atol=0.05)


def test_quantize_3d_transformer_hidden():
    # 3D Activation: (batch_size, seq_len, hidden_dim)
    activations = torch.randn(4, 32, 128)
    q, scale, zp = quantize_per_tensor(activations, bits=8, symmetric=False)

    recon = dequantize_asymmetric(q, scale, zp)
    assert recon.shape == activations.shape
    assert torch.allclose(activations, recon, atol=0.08)


def test_quantize_grouped_odd_remainder():
    # 100 elements is not divisible by group_size 32 (32 * 3 = 96, 4 remainder)
    x = torch.randn(10, 10)
    assert x.numel() == 100

    q, scale, zp, orig_shape = quantize_grouped(x, group_size=32, bits=4, symmetric=True)
    recon = dequantize_grouped(q, scale, zp, orig_shape)

    assert recon.shape == (10, 10)
    assert not torch.isnan(recon).any()


def test_calibrate_percentile_extreme():
    x = torch.randn(10000)
    # 99.999th percentile
    low, high = calibrate_percentile(x, percentile=99.999)
    assert high.item() > low.item()


def test_calibrate_mse_preserves_bounds():
    x = torch.randn(64, 64)
    res = calibrate_mse(x, num_candidates=15, bits=4)

    assert 0.69 <= res["best_alpha"] <= 1.00
    assert res["clipped_tensor"].shape == x.shape
