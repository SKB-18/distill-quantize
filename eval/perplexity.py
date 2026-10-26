"""
Perplexity Evaluation for Autoregressive Language Models.
"""

from typing import Optional, Union, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    device: Optional[str] = None
) -> float:
    """
    Computes Perplexity (PPL = exp(loss)) of a causal language model on a sequence.
    
    Args:
        model: Autoregressive language model.
        input_ids: (batch_size, seq_len) token IDs.
        labels: Optional targets (defaults to shifted input_ids).
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    input_ids = input_ids.to(dev)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if hasattr(outputs, "logits"):
            logits = outputs.logits

        # Shift logits and labels for next-token prediction
        # logits[:, :-1, :] predicts labels[:, 1:]
        shift_logits = logits[..., :-1, :].contiguous()
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous().to(dev)
        else:
            shift_labels = input_ids[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean"
        )
        
        ppl = math.exp(min(loss.item(), 100.0))
        return float(ppl)
