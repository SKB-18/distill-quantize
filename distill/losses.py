"""
Hinton Knowledge Distillation and Sequence-Level KL Divergence Losses.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class HintonKDLoss(nn.Module):
    """
    Classical Hinton Knowledge Distillation Loss (Hinton et al., 2015).
    
    Combines temperature-scaled Kullback-Leibler (KL) divergence over teacher soft
    probabilities with standard cross-entropy over ground-truth hard labels:
    
        L_total = alpha * (T^2 * KL(sigma(z_s / T) || sigma(z_t / T))) + (1 - alpha) * CE(z_s, y)
        
    The T^2 scaling factor ensures that the gradients produced by the soft targets
    scale proportionally with temperature, matching the scale of the cross-entropy loss.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        reduction: str = "batchmean"
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be between 0.0 and 1.0, got {alpha}")
            
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.reduction = reduction
        self.kl_div = nn.KLDivLoss(reduction=reduction)
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates combined KD loss.
        
        Args:
            student_logits: Tensor of shape (..., num_classes)
            teacher_logits: Tensor of shape (..., num_classes)
            labels: Optional ground-truth integer labels of shape (...)
            
        Returns:
            Scalar loss tensor.
        """
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"Student shape {student_logits.shape} does not match teacher shape {teacher_logits.shape}"
            )

        # Reshape multidimensional logits (e.g., sequence generation) into 2D (N, C)
        s_logits_flat = student_logits.view(-1, student_logits.size(-1))
        t_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))

        # Soft target distillation: log_softmax for student, softmax for teacher
        s_soft = F.log_softmax(s_logits_flat / self.temperature, dim=-1)
        t_soft = F.softmax(t_logits_flat / self.temperature, dim=-1)

        # PyTorch KLDivLoss expects input=log_probs, target=probs
        soft_loss = self.kl_div(s_soft, t_soft) * (self.temperature ** 2)

        # Hard target loss
        if labels is not None and self.alpha < 1.0:
            labels_flat = labels.view(-1)
            hard_loss = self.cross_entropy(s_logits_flat, labels_flat)
            return self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss
        
        return soft_loss


class SequenceKDLoss(nn.Module):
    """
    Token-level Sequence Knowledge Distillation with Attention Mask Support.
    
    Applies KL divergence only over valid non-padded tokens in autoregressive sequences.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.7,
        ignore_index: int = -100
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            student_logits: (batch, seq_len, vocab_size)
            teacher_logits: (batch, seq_len, vocab_size)
            labels: Optional (batch, seq_len)
            attention_mask: Optional (batch, seq_len) with 1 for tokens, 0 for pad
        """
        batch_size, seq_len, vocab_size = student_logits.shape
        
        # Soft token distributions
        s_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        t_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Pointwise KL: sum(P * (log(P) - log(Q)))
        # Here input is s_log_probs (log Q), target is t_probs (P).
        # F.kl_div(s_log_probs, t_probs, reduction='none') = P * (log P - log Q)
        token_kl = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)  # (batch, seq_len)
        token_kl = token_kl * (self.temperature ** 2)

        if attention_mask is not None:
            mask = attention_mask.bool()
            valid_tokens = mask.sum().clamp(min=1)
            soft_loss = (token_kl * mask).sum() / valid_tokens
        else:
            soft_loss = token_kl.mean()

        if labels is not None and self.alpha < 1.0:
            hard_loss = F.cross_entropy(
                student_logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=self.ignore_index,
                reduction="mean"
            )
            return self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss

        return soft_loss
