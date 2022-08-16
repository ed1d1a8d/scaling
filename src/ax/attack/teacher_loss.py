import torch
import torch.nn.functional as F
from torch import nn


class TeacherCrossEntropy:
    def __init__(self, teacher: nn.Module):
        """Teacher should return logits."""
        self.teacher = teacher

    def get_loss(self, inputs: torch.Tensor, student_logits: torch.Tensor):
        teacher_logits: torch.Tensor = self.teacher(inputs)
        teacher_probs = teacher_logits.softmax(dim=-1)
        return F.cross_entropy(input=student_logits, target=teacher_probs)
