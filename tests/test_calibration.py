"""
Unit Tests for Calibration Algorithms (MinMax, Percentile, MSE Grid Search).
"""

import pytest
import torch
from quantize.calibration import calibrate_minmax, calibrate_percentile, calibrate_mse


def test_calibrate_minmax():
    x = torch.tensor([-5.5, 0.0, 12.3])
    low, high = calibrate_minmax(x)
    assert abs(low.item() - (-5.5)) < 1e-4
    assert abs(high.item() - 12.3) < 1e-4


def test_calibrate_percentile_outlier_suppression():
    # Mostly values in [-1, 1], with one extreme outlier at 500
    x = torch.randn(1000)
    x[0] = 500.0
    x[1] = -500.0

    low, high = calibrate_percentile(x, percentile=99.0)
    assert high.item() < 100.0  # Outlier 500 was clipped
    assert low.item() > -100.0


def test_calibrate_percentile_invalid_range():
    x = torch.randn(10)
    with pytest.raises(ValueError, match="between 50.0 and 100.0"):
        calibrate_percentile(x, percentile=45.0)


def test_calibrate_mse_search():
    x = torch.randn(32, 32)
    # Inject large outlier
    x[0, 0] = 20.0

    res = calibrate_mse(x, num_candidates=10, bits=8, symmetric=True)
    assert "best_alpha" in res
    assert "best_mse" in res
    assert 0.7 <= res["best_alpha"] <= 1.0
    assert res["best_mse"] >= 0.0


def test_calibrate_mse_zero_tensor():
    x = torch.zeros(10, 10)
    res = calibrate_mse(x, bits=8)
    assert res["best_alpha"] == 1.0
    assert res["best_mse"] == 0.0
