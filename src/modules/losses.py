import torch.nn as nn
import torch.nn.functional as F
import torchvision

class BCEFocalLoss(nn.Module):
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2.0,
            reduction: str = "mean",
            bce_weight: float = 0.6,
            focal_weight: float = 1.4,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        targets = targets.float()

        focal_loss = torchvision.ops.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        
        bce_loss = self.bce(logits, targets)
        
        combined_loss = self.bce_weight * bce_loss + self.focal_weight * focal_loss

        return combined_loss
