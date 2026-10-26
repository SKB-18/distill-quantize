"""
Quantization Granularities: Per-Tensor, Per-Channel, and Grouped / Block-wise.
"""

from typing import Tuple, Optional
import torch
from .numerics import get_qmin_qmax


def quantize_per_tensor(
    x: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantizes an entire tensor using a single global scale (and zero point).
    """
    qmin, qmax = get_qmin_qmax(bits, symmetric)
    
    if symmetric:
        max_val = torch.max(torch.abs(x))
        scale = torch.clamp(max_val / float(qmax), min=eps)
        q = torch.clamp(torch.round(x / scale), qmin, qmax)
        zero_point = None
    else:
        min_val = torch.min(x)
        max_val = torch.max(x)
        min_val = torch.minimum(min_val, torch.tensor(0.0, dtype=x.dtype, device=x.device))
        max_val = torch.maximum(max_val, torch.tensor(0.0, dtype=x.dtype, device=x.device))
        val_range = torch.clamp(max_val - min_val, min=eps)
        scale = val_range / float(qmax - qmin)
        zero_point = torch.clamp(torch.round(-min_val / scale) + float(qmin), qmin, qmax)
        q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
        
    return q.to(torch.int8 if symmetric else torch.uint8), scale, zero_point


def quantize_per_channel(
    x: torch.Tensor,
    dim: int = 0,
    bits: int = 8,
    symmetric: bool = True,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantizes a 2D or multi-D weight tensor with independent scale per slice along dim.
    Typically used for Linear layer weights (dim=0 corresponds to output channels/rows).
    """
    qmin, qmax = get_qmin_qmax(bits, symmetric)
    shape = x.shape
    
    # Target reduction dimensions (all dimensions except dim)
    reduce_dims = [d for d in range(x.ndim) if d != dim]
    
    if symmetric:
        max_val = torch.amax(torch.abs(x), dim=reduce_dims, keepdim=True)
        scale = torch.clamp(max_val / float(qmax), min=eps)
        q = torch.clamp(torch.round(x / scale), qmin, qmax)
        zero_point = None
    else:
        min_val = torch.amin(x, dim=reduce_dims, keepdim=True)
        max_val = torch.amax(x, dim=reduce_dims, keepdim=True)
        zero_tensor = torch.zeros_like(min_val)
        min_val = torch.minimum(min_val, zero_tensor)
        max_val = torch.maximum(max_val, zero_tensor)
        val_range = torch.clamp(max_val - min_val, min=eps)
        scale = val_range / float(qmax - qmin)
        zero_point = torch.clamp(torch.round(-min_val / scale) + float(qmin), qmin, qmax)
        q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)

    return q.to(torch.int8 if symmetric else torch.uint8), scale, zero_point


def quantize_grouped(
    x: torch.Tensor,
    group_size: int = 32,
    bits: int = 4,
    symmetric: bool = True,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Tuple[int, ...]]:
    """
    Block-wise / Grouped Quantization (as used in AWQ, GPTQ, GGUF).
    
    Divides weight rows into blocks of size `group_size` and computes individual
    scales for each sub-block.
    
    Returns:
        (quantized_flat, scales, zero_points, original_shape)
    """
    original_shape = x.shape
    num_elements = x.numel()
    
    if num_elements % group_size != 0:
        # Pad to multiple of group_size if necessary
        pad_len = group_size - (num_elements % group_size)
        x_flat = torch.cat([x.reshape(-1), torch.zeros(pad_len, dtype=x.dtype, device=x.device)])
    else:
        x_flat = x.reshape(-1)

    # Reshape into (N_groups, group_size)
    grouped = x_flat.view(-1, group_size)
    qmin, qmax = get_qmin_qmax(bits, symmetric)
    
    if symmetric:
        max_val = torch.amax(torch.abs(grouped), dim=-1, keepdim=True)
        scale = torch.clamp(max_val / float(qmax), min=eps)
        q = torch.clamp(torch.round(grouped / scale), qmin, qmax)
        zero_point = None
    else:
        min_val = torch.amin(grouped, dim=-1, keepdim=True)
        max_val = torch.amax(grouped, dim=-1, keepdim=True)
        val_range = torch.clamp(max_val - min_val, min=eps)
        scale = val_range / float(qmax - qmin)
        zero_point = torch.clamp(torch.round(-min_val / scale) + float(qmin), qmin, qmax)
        q = torch.clamp(torch.round(grouped / scale) + zero_point, qmin, qmax)

    return q.to(torch.int8 if symmetric else torch.uint8), scale, zero_point, original_shape


def dequantize_grouped(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    original_shape: Tuple[int, ...],
    target_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Dequantizes a grouped representation back to its original tensor shape.
    """
    if zero_point is not None:
        dequant = (q.to(target_dtype) - zero_point.to(target_dtype)) * scale
    else:
        dequant = q.to(target_dtype) * scale
        
    flat = dequant.reshape(-1)
    num_orig = 1
    for d in original_shape:
        num_orig *= d
        
    return flat[:num_orig].view(original_shape)
