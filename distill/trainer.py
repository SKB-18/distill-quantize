"""
Knowledge Distillation Training and Evaluation Loop.
"""

from typing import Dict, Any, Optional, Iterator, Tuple
import torch
import torch.nn as nn
from .losses import HintonKDLoss
from .feature_kd import FeatureProjectionLoss


class DistillationTrainer:
    """
    Standardized trainer orchestrating student distillation from a frozen teacher.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        optimizer: torch.optim.Optimizer,
        kd_loss: Optional[HintonKDLoss] = None,
        feature_loss: Optional[FeatureProjectionLoss] = None,
        feature_weight: float = 0.1,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device)
        
        # Teacher must be strictly frozen in evaluation mode
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        self.optimizer = optimizer
        self.kd_loss = kd_loss or HintonKDLoss(temperature=4.0, alpha=0.5)
        self.feature_loss = feature_loss
        self.feature_weight = feature_weight
        
        # Telemetry
        self.step_count = 0
        self.loss_history = []

    def train_step(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Executes a single optimization step.
        """
        self.student.train()
        self.optimizer.zero_grad()
        
        inputs = inputs.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        # 1. Teacher forward (no gradient graph)
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)
            if isinstance(teacher_outputs, tuple):
                t_logits = teacher_outputs[0]
                t_hidden = teacher_outputs[1] if len(teacher_outputs) > 1 else None
            else:
                t_logits = teacher_outputs
                t_hidden = None

        # 2. Student forward
        student_outputs = self.student(inputs)
        if isinstance(student_outputs, tuple):
            s_logits = student_outputs[0]
            s_hidden = student_outputs[1] if len(student_outputs) > 1 else None
        else:
            s_logits = student_outputs
            s_hidden = None

        # 3. Compute loss
        loss = self.kd_loss(s_logits, t_logits, labels)
        feat_loss_val = 0.0

        if self.feature_loss is not None and s_hidden is not None and t_hidden is not None:
            f_loss = self.feature_loss(s_hidden, t_hidden)
            loss = loss + self.feature_weight * f_loss
            feat_loss_val = float(f_loss.item())

        loss.backward()
        self.optimizer.step()
        self.step_count += 1
        
        loss_val = float(loss.item())
        self.loss_history.append(loss_val)

        return {
            "step": self.step_count,
            "total_loss": loss_val,
            "feature_loss": feat_loss_val
        }

    def evaluate(
        self,
        dataloader: Iterator[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Evaluates student performance on test/val set.
        """
        self.student.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.student(inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
                loss = nn.functional.cross_entropy(logits, targets)
                total_loss += float(loss.item())
                batches += 1
                
                preds = logits.argmax(dim=-1)
                total_correct += int((preds == targets).sum().item())
                total_samples += targets.size(0)

        accuracy = (total_correct / total_samples) if total_samples > 0 else 0.0
        avg_loss = (total_loss / batches) if batches > 0 else 0.0
        
        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "total_samples": total_samples
        }
