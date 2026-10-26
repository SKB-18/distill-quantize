"""
Unit Tests for QuantizedLinear Neural Layer.
"""

import io
import torch
import torch.nn as nn
from layers.quantized_linear import QuantizedLinear
from eval.metrics import compute_cosine_similarity


def test_quantized_linear_from_float_symmetric():
    float_linear = nn.Linear(32, 64, bias=True)
    q_linear = QuantizedLinear.from_float(
        float_linear, bits=8, symmetric=True, granularity="per_channel"
    )

    x = torch.randn(8, 32)
    y_float = float_linear(x)
    y_quant = q_linear(x)

    assert y_quant.shape == (8, 64)
    # Cosine similarity between original and quantized layer should be very high (>0.999)
    cos = compute_cosine_similarity(y_float, y_quant)
    assert cos > 0.999


def test_quantized_linear_from_float_asymmetric():
    float_linear = nn.Linear(16, 32, bias=True)
    q_linear = QuantizedLinear.from_float(
        float_linear, bits=8, symmetric=False, granularity="per_channel"
    )

    x = torch.randn(4, 16)
    y_float = float_linear(x)
    y_quant = q_linear(x)

    assert y_quant.shape == (4, 32)
    cos = compute_cosine_similarity(y_float, y_quant)
    assert cos > 0.99


def test_quantized_linear_no_bias():
    float_linear = nn.Linear(20, 10, bias=False)
    q_linear = QuantizedLinear.from_float(float_linear, bits=8)
    assert q_linear.bias is None

    x = torch.randn(5, 20)
    out = q_linear(x)
    assert out.shape == (5, 10)


def test_quantized_linear_state_dict_save_load():
    float_linear = nn.Linear(16, 16)
    q1 = QuantizedLinear.from_float(float_linear, bits=8)

    buf = io.BytesIO()
    torch.save(q1.state_dict(), buf)
    buf.seek(0)

    q2 = QuantizedLinear(16, 16, bits=8)
    q2.load_state_dict(torch.load(buf))

    x = torch.randn(2, 16)
    assert torch.equal(q1(x), q2(x))


def test_quantized_linear_memory_reduction():
    in_f, out_f = 512, 1024
    float_linear = nn.Linear(in_f, out_f, bias=False)
    q_linear = QuantizedLinear.from_float(float_linear, bits=8, granularity="per_channel")

    fp32_bytes = in_f * out_f * 4
    int8_bytes = q_linear.memory_bytes()
    ratio = fp32_bytes / int8_bytes

    # Theoretical reduction should be between 3.9x and 4.0x
    assert 3.8 <= ratio <= 4.1
