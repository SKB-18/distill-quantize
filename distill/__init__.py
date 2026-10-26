"""
Knowledge Distillation Engine
Implements Hinton logit matching, temperature scaling, and intermediate feature projection.
"""

from .losses import HintonKDLoss, SequenceKDLoss
from .feature_kd import FeatureProjectionLoss
from .trainer import DistillationTrainer

__all__ = [
    "HintonKDLoss",
    "SequenceKDLoss",
    "FeatureProjectionLoss",
    "DistillationTrainer",
]
