"""
Unit Tests for Distortion and Perplexity Evaluation Metrics.
"""

import math
import torch
import torch.nn as nn
from eval.metrics import (
    compute_sqnr,
    compute_cosine_similarity,
    compute_mae,
    compute_rmse,
    evaluate_quantization_distortion,
)
from eval.perplexity import compute_perplexity


def test_sqnr_identical_tensors():
    x = torch.randn(10, 10)
    sqnr = compute_sqnr(x, x.clone())
    assert math.isinf(sqnr)


def test_sqnr_noisy_tensor():
    x = torch.ones(100)
    noise = torch.ones(100) * 0.1
    # signal power = 100 * 1.0 = 100
    # noise power = 100 * 0.01 = 1.0
    # ratio = 100 -> 10 * log10(100) = 20.0 dB
    sqnr = compute_sqnr(x, x + noise)
    assert math.isclose(sqnr, 20.0, rel_tol=1e-3)


def test_cosine_similarity():
    x = torch.tensor([1.0, 0.0, 0.0])
    y = torch.tensor([0.0, 1.0, 0.0])
    assert math.isclose(compute_cosine_similarity(x, y), 0.0, abs_tol=1e-5)

    same = compute_cosine_similarity(x, x)
    assert math.isclose(same, 1.0, abs_tol=1e-5)


def test_mae_and_rmse():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([2.0, 2.0, 5.0])
    assert math.isclose(compute_mae(x, y), 1.0, abs_tol=1e-5)
    # diff = [1, 0, 2], squared = [1, 0, 4], mean = 5/3, sqrt = sqrt(5/3) = 1.29099
    assert math.isclose(compute_rmse(x, y), math.sqrt(5.0 / 3.0), abs_tol=1e-5)


class MockLM(nn.Module):
    def __init__(self, vocab_size=50):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32)
        self.head = nn.Linear(32, vocab_size)

    def forward(self, input_ids):
        return self.head(self.embed(input_ids))


def test_perplexity():
    model = MockLM(vocab_size=100)
    input_ids = torch.randint(0, 100, (2, 16))
    ppl = compute_perplexity(model, input_ids)
    assert ppl > 1.0
    assert not math.isnan(ppl)
