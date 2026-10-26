"""
Unit Tests for PTQ Quantization Numerics and Granularities.
"""

import pytest
import torch
from quantize.numerics import (
    quantize_symmetric,
    dequantize_symmetric,
    quantize_asymmetric,
    dequantize_asymmetric,
    get_qmin_qmax,
)
from quantize.granularity import (
    quantize_per_tensor,
    quantize_per_channel,
    quantize_grouped,
    dequantize_grouped,
)


def test_qmin_qmax_bounds():
    s_min8, s_max8 = get_qmin_qmax(8, symmetric=True)
    assert s_min8 == -128 and s_max8 == 127

    s_min4, s_max4 = get_qmin_qmax(4, symmetric=True)
    assert s_min4 == -8 and s_max4 == 7

    u_min8, u_max8 = get_qmin_qmax(8, symmetric=False)
    assert u_min8 == 0 and u_max8 == 255

    u_min4, u_max4 = get_qmin_qmax(4, symmetric=False)
    assert u_min4 == 0 and u_max4 == 15


def test_quantize_symmetric_int8():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    q, scale = quantize_symmetric(x, bits=8)

    assert q.dtype == torch.int8
    assert q[2].item() == 0  # Zero must map exactly to integer 0
    assert q[0].item() == -127 or q[0].item() == -128
    assert q[4].item() == 127

    # Roundtrip
    recon = dequantize_symmetric(q, scale)
    assert torch.allclose(x, recon, atol=0.03)


def test_quantize_asymmetric_int8():
    x = torch.tensor([10.0, 15.0, 20.0, 25.0])
    q, scale, zp = quantize_asymmetric(x, bits=8)

    assert q.dtype == torch.uint8
    assert q.min().item() >= 0
    assert q.max().item() <= 255

    recon = dequantize_asymmetric(q, scale, zp)
    assert torch.allclose(x, recon, atol=0.1)


def test_quantize_int4_bounds():
    x = torch.randn(10, 10)
    q, scale = quantize_symmetric(x, bits=4)
    assert q.min().item() >= -8
    assert q.max().item() <= 7


def test_per_channel_isolated_scales():
    # Row 0 has huge values, Row 1 has tiny values
    x = torch.tensor([
        [100.0, -100.0, 50.0],
        [0.01, -0.01, 0.005]
    ])

    # Per-channel along dim 0
    q, scale, _ = quantize_per_channel(x, dim=0, bits=8, symmetric=True)
    assert scale.shape == (2, 1)
    assert scale[0].item() > scale[1].item() * 1000

    recon = q.float() * scale
    assert torch.allclose(x[1], recon[1], atol=1e-3)


def test_grouped_quantization_roundtrip():
    x = torch.randn(16, 64)
    q, scale, zp, orig_shape = quantize_grouped(x, group_size=32, bits=4, symmetric=True)
    recon = dequantize_grouped(q, scale, zp, orig_shape)

    assert recon.shape == x.shape
    # Error should be bounded for INT4
    assert torch.mean(torch.abs(x - recon)).item() < 0.35


def test_all_zeros_tensor_safety():
    x = torch.zeros(5, 5)
    q, scale = quantize_symmetric(x, bits=8)
    assert not torch.isnan(scale).any()
    assert torch.all(q == 0)
