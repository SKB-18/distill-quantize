"""
Calibration Routines: MinMax, Percentile Outlier Clipping, and Optimal MSE Grid Search.
"""

from typing import Tuple, Dict, Any
import torch
from .numerics import quantize_symmetric, dequantize_symmetric


def calibrate_minmax(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard MinMax calibration preserving true tensor extrema.
    """
    return torch.min(x), torch.max(x)


def calibrate_percentile(
    x: torch.Tensor,
    percentile: float = 99.99
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Clips extreme outliers beyond specified percentile.
    
    Args:
        x: Input tensor
        percentile: Float in [50.0, 100.0] representing the upper quantile.
    """
    if not (50.0 <= percentile <= 100.0):
        raise ValueError(f"Percentile must be between 50.0 and 100.0, got {percentile}")
        
    x_flat = x.detach().reshape(-1).float()
    q_high = percentile / 100.0
    q_low = 1.0 - q_high

    upper = torch.quantile(x_flat, q_high)
    lower = torch.quantile(x_flat, q_low)
    
    return lower, upper


def calibrate_mse(
    x: torch.Tensor,
    num_candidates: int = 25,
    bits: int = 8,
    symmetric: bool = True
) -> Dict[str, Any]:
    """
    Searches for the optimal clipping factor alpha in [0.70, 1.0] that minimizes
    the Mean Squared Error (MSE) between the original unclipped tensor and the
    dequantized reconstruction:
    
        alpha* = argmin_{alpha} || x - dequantize(quantize(clamp(x, -alpha*M, alpha*M))) ||_2^2
        
    Balances clipping distortion against quantization bin resolution.
    """
    max_val = torch.max(torch.abs(x)).item()
    if max_val == 0.0:
        return {
            "best_alpha": 1.0,
            "best_mse": 0.0,
            "clipped_tensor": x
        }

    best_alpha = 1.0
    best_mse = float("inf")
    best_clipped = x

    # Search space from 70% to 100% of maximum magnitude
    alphas = torch.linspace(0.70, 1.0, steps=num_candidates)

    for alpha_t in alphas:
        alpha = float(alpha_t.item())
        threshold = alpha * max_val
        
        # Clip
        clipped = torch.clamp(x, -threshold, threshold)
        
        # Quantize and dequantize
        q, scale = quantize_symmetric(clipped, bits=bits)
        reconstructed = dequantize_symmetric(q, scale, target_dtype=x.dtype)
        
        # Compute MSE against ORIGINAL unclipped tensor
        mse = torch.mean((x - reconstructed) ** 2).item()
        
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_clipped = clipped

    return {
        "best_alpha": best_alpha,
        "best_mse": best_mse,
        "clipped_tensor": best_clipped
    }
