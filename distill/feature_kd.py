"""
Intermediate Feature and Hidden-State Distillation (FitNets style).
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjectionLoss(nn.Module):
    """
    Intermediate Feature Distillation with Learnable Projection.
    
    When student hidden state dimension d_s differs from teacher dimension d_t,
    a linear transformation W_p projects student representations into the teacher
    manifold:
        L_feat = 1/2 * || W_p(h_s) - h_t ||_2^2
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        criterion: str = "mse"
    ):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.criterion = criterion.lower()

        if student_dim != teacher_dim:
            self.projector = nn.Linear(student_dim, teacher_dim, bias=False)
            # Initialize with identity-like orthogonal weights
            nn.init.orthogonal_(self.projector.weight)
        else:
            self.projector = nn.Identity()

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates projection matching loss between student and teacher representations.
        
        Args:
            student_features: Tensor of shape (batch, ..., student_dim)
            teacher_features: Tensor of shape (batch, ..., teacher_dim)
        """
        proj_student = self.projector(student_features)
        
        if self.criterion == "mse":
            return F.mse_loss(proj_student, teacher_features)
        elif self.criterion == "cosine":
            # 1 - cosine_similarity
            cos = F.cosine_similarity(proj_student, teacher_features, dim=-1)
            return (1.0 - cos).mean()
        elif self.criterion == "smooth_l1":
            return F.smooth_l1_loss(proj_student, teacher_features)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")


class MultiLayerFeatureKDLoss(nn.Module):
    """
    Distills multiple intermediate hidden layers simultaneously.
    """

    def __init__(
        self,
        student_dims: List[int],
        teacher_dims: List[int],
        layer_weights: Optional[List[float]] = None
    ):
        super().__init__()
        if len(student_dims) != len(teacher_dims):
            raise ValueError("student_dims and teacher_dims must have the same length")
            
        self.projectors = nn.ModuleList([
            FeatureProjectionLoss(s_dim, t_dim)
            for s_dim, t_dim in zip(student_dims, teacher_dims)
        ])
        
        if layer_weights is None:
            self.layer_weights = [1.0 / len(student_dims)] * len(student_dims)
        else:
            total = sum(layer_weights)
            self.layer_weights = [w / total for w in layer_weights]

    def forward(
        self,
        student_hidden_states: List[torch.Tensor],
        teacher_hidden_states: List[torch.Tensor]
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=student_hidden_states[0].device)
        for proj, weight, s_h, t_h in zip(
            self.projectors, self.layer_weights, student_hidden_states, teacher_hidden_states
        ):
            total_loss = total_loss + weight * proj(s_h, t_h)
        return total_loss
