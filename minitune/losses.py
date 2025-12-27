# minitune/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [Batch, Sequence_Len, Vocab_Size]
        # labels: [Batch, Sequence_Len]
        
        # Flatten for calculation
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        
        # Standard Cross Entropy
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # Calculate prob (p_t)
        pt = torch.exp(-ce_loss)
        
        # Focal term
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss