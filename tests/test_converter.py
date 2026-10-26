"""
Unit Tests for Recursive Model Architecture Converter.
"""

import torch
import torch.nn as nn
from layers.converter import convert_to_quantized
from layers.quantized_linear import QuantizedLinear
from eval.metrics import compute_cosine_similarity


class MultiLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        return self.classifier(self.act(self.fc2(self.act(self.fc1(x)))))


def test_convert_recursive():
    model = MultiLayerModel()
    q_model, stats = convert_to_quantized(model, bits=8, exclude_names=[])

    assert isinstance(q_model.fc1, QuantizedLinear)
    assert isinstance(q_model.fc2, QuantizedLinear)
    assert isinstance(q_model.classifier, QuantizedLinear)
    assert stats["total_converted"] == 3
    assert stats["compression_ratio"] > 1.5


def test_convert_with_exclusions():
    model = MultiLayerModel()
    q_model, stats = convert_to_quantized(model, bits=8, exclude_names=["classifier"])

    assert isinstance(q_model.fc1, QuantizedLinear)
    assert isinstance(q_model.fc2, QuantizedLinear)
    assert isinstance(q_model.classifier, nn.Linear)  # Kept as float!
    assert stats["total_converted"] == 2
    assert "classifier" in stats["skipped_layers"]


def test_converted_model_numerical_fidelity():
    model = MultiLayerModel()
    q_model, _ = convert_to_quantized(model, bits=8)

    x = torch.randn(8, 32)
    y_orig = model(x)
    y_quant = q_model(x)

    assert y_quant.shape == (8, 10)
    cos = compute_cosine_similarity(y_orig, y_quant)
    assert cos > 0.99
