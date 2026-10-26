"""
Recursive Architecture Converter: replaces standard Linear layers with QuantizedLinear.
"""

from typing import List, Optional, Dict, Any, Tuple
import copy
import torch
import torch.nn as nn
from .quantized_linear import QuantizedLinear


def get_model_size_bytes(model: nn.Module) -> int:
    """Calculates total memory footprint of all parameters and buffers in bytes."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes


def convert_to_quantized(
    model: nn.Module,
    bits: int = 8,
    symmetric: bool = True,
    granularity: str = "per_channel",
    calibration: str = "minmax",
    exclude_names: Optional[List[str]] = None,
    inplace: bool = False
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Recursively replaces all eligible nn.Linear layers in a model with QuantizedLinear.
    
    Args:
        model: PyTorch model.
        bits: 8 or 4.
        symmetric: True for symmetric, False for asymmetric.
        granularity: "per_channel" or "per_tensor".
        calibration: "minmax" or "mse".
        exclude_names: Substrings or names of layers to skip (e.g. ['head', 'classifier']).
        inplace: If False, clones the model first.
        
    Returns:
        (quantized_model, conversion_stats)
    """
    exclude = exclude_names if exclude_names is not None else ["lm_head", "classifier"]
    target_model = model if inplace else copy.deepcopy(model)
    
    orig_bytes = get_model_size_bytes(target_model)
    converted_layers: List[str] = []
    skipped_layers: List[str] = []

    def _replace_recursive(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check exclusions
            is_excluded = any(excl in full_name for excl in exclude)
            
            if isinstance(child, nn.Linear):
                if is_excluded:
                    skipped_layers.append(full_name)
                else:
                    q_layer = QuantizedLinear.from_float(
                        child,
                        bits=bits,
                        symmetric=symmetric,
                        granularity=granularity,
                        calibration=calibration
                    )
                    setattr(module, name, q_layer)
                    converted_layers.append(full_name)
            else:
                _replace_recursive(child, full_name)

    _replace_recursive(target_model)
    
    quant_bytes = get_model_size_bytes(target_model)
    compression_ratio = orig_bytes / max(quant_bytes, 1)

    stats = {
        "converted_layers": converted_layers,
        "skipped_layers": skipped_layers,
        "total_converted": len(converted_layers),
        "orig_size_mb": orig_bytes / (1024 * 1024),
        "quant_size_mb": quant_bytes / (1024 * 1024),
        "compression_ratio": compression_ratio,
        "bits": bits,
        "symmetric": symmetric,
        "granularity": granularity
    }

    return target_model, stats


Tuple_QuantizedResult = Any
