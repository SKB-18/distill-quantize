"""
Quantization Distortion Metrics: SQNR, Cosine Similarity, MAE, and RMSE.
"""

from typing import Dict, Any
import math
import torch
import torch.nn.functional as F


def compute_sqnr(original: torch.Tensor, reconstructed: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Computes Signal-to-Quantization-Noise Ratio (SQNR) in decibels (dB):
        SQNR = 10 * log10 ( ||x||_2^2 / ||x - x_hat||_2^2 )
        
    High SQNR (>40 dB) indicates virtually indistinguishable numerical fidelity.
    """
    signal_power = torch.sum(original.float() ** 2).item()
    noise_power = torch.sum((original.float() - reconstructed.float()) ** 2).item()

    if noise_power <= eps:
        return float("inf")
    if signal_power <= eps:
        return 0.0

    sqnr_db = 10.0 * math.log10(signal_power / noise_power)
    return float(sqnr_db)


def compute_cosine_similarity(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Computes average cosine similarity between original and reconstructed tensors.
    """
    o_flat = original.reshape(-1).float()
    r_flat = reconstructed.reshape(-1).float()
    cos = F.cosine_similarity(o_flat.unsqueeze(0), r_flat.unsqueeze(0)).item()
    return float(cos)


def compute_mae(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return float(torch.mean(torch.abs(original.float() - reconstructed.float())).item())


def compute_rmse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Root Mean Square Error."""
    return float(torch.sqrt(torch.mean((original.float() - reconstructed.float()) ** 2)).item())


def evaluate_quantization_distortion(
    original: torch.Tensor,
    reconstructed: torch.Tensor
) -> Dict[str, float]:
    """
    Computes comprehensive distortion report between original and reconstructed tensors.
    """
    return {
        "sqnr_db": compute_sqnr(original, reconstructed),
        "cosine_similarity": compute_cosine_similarity(original, reconstructed),
        "mae": compute_mae(original, reconstructed),
        "rmse": compute_rmse(original, reconstructed)
    }
