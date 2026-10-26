"""
Core Quantization Numerics: Symmetric and Asymmetric Roundtrip Logic.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class QuantParams:
    """Encapsulates quantization metadata and scale/offset tensors."""
    scale: torch.Tensor
    zero_point: Optional[torch.Tensor]
    bits: int
    symmetric: bool
    qmin: int
    qmax: int


def get_qmin_qmax(bits: int, symmetric: bool) -> Tuple[int, int]:
    """
    Returns the integer bounds for a given bitwidth and symmetry mode.
    
    Signed symmetric:
      - 8-bit: [-128, 127] (or full range [-128, 127] with clamp)
      - 4-bit: [-8, 7]
    Unsigned asymmetric:
      - 8-bit: [0, 255]
      - 4-bit: [0, 15]
    """
    if bits <= 0 or bits > 32:
        raise ValueError(f"Bits must be in range [1, 32], got {bits}")
        
    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        qmin = -(1 << (bits - 1))
    else:
        qmin = 0
        qmax = (1 << bits) - 1
        
    return qmin, qmax


def quantize_symmetric(
    x: torch.Tensor,
    bits: int = 8,
    qmin: Optional[int] = None,
    qmax: Optional[int] = None,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric uniform quantization centered at 0:
        scale = max(|x|) / qmax
        q = clamp(round(x / scale), qmin, qmax)
        
    Returns:
        (quantized_tensor, scale)
    """
    default_qmin, default_qmax = get_qmin_qmax(bits, symmetric=True)
    qmin = qmin if qmin is not None else default_qmin
    qmax = qmax if qmax is not None else default_qmax

    max_val = torch.max(torch.abs(x))
    scale = torch.clamp(max_val / float(qmax), min=eps)

    # Round to nearest integer and clamp
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    
    # Store in native integer dtype if standard bitwidths
    if bits == 8:
        q = q.to(torch.int8)
    elif bits <= 4:
        q = q.to(torch.int8)  # Representable in int8 container
        
    return q, scale


def dequantize_symmetric(
    q: torch.Tensor,
    scale: torch.Tensor,
    target_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Dequantizes symmetric integer tensor:
        x_hat = q * scale
    """
    return q.to(target_dtype) * scale


def quantize_asymmetric(
    x: torch.Tensor,
    bits: int = 8,
    qmin: Optional[int] = None,
    qmax: Optional[int] = None,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric uniform quantization with explicit zero-point offset:
        scale = (max(x) - min(x)) / (qmax - qmin)
        zero_point = round(-min(x) / scale) + qmin
        q = clamp(round(x / scale) + zero_point, qmin, qmax)
        
    Returns:
        (quantized_tensor, scale, zero_point)
    """
    default_qmin, default_qmax = get_qmin_qmax(bits, symmetric=False)
    qmin = qmin if qmin is not None else default_qmin
    qmax = qmax if qmax is not None else default_qmax

    min_val = torch.min(x)
    max_val = torch.max(x)

    # Ensure real 0.0 is representable in asymmetric grid
    min_val = torch.minimum(min_val, torch.tensor(0.0, dtype=x.dtype, device=x.device))
    max_val = torch.maximum(max_val, torch.tensor(0.0, dtype=x.dtype, device=x.device))

    val_range = torch.clamp(max_val - min_val, min=eps)
    scale = val_range / float(qmax - qmin)

    # Zero-point calculation
    initial_zero_point = -min_val / scale + float(qmin)
    zero_point = torch.clamp(torch.round(initial_zero_point), qmin, qmax)

    # Quantize
    q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    
    if bits == 8:
        q = q.to(torch.uint8)
    elif bits <= 4:
        q = q.to(torch.uint8)
        
    return q, scale, zero_point


def dequantize_asymmetric(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    target_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Dequantizes asymmetric integer tensor:
        x_hat = (q - zero_point) * scale
    """
    return (q.to(target_dtype) - zero_point.to(target_dtype)) * scale
