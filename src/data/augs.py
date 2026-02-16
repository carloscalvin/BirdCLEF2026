import numpy as np
import torch
import torch.nn as nn

class Mixup(nn.Module):
    def __init__(self, mixup_prob=0.5, alpha=1.0):
        super().__init__()
        self.mixup_prob = mixup_prob
        self.alpha = alpha

    def forward(self, x, y):
        if self.mixup_prob <= 0 or np.random.rand() > self.mixup_prob:
            return x, y

        batch_size = x.size(0)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        lam = max(lam, 1. - lam) 
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]

        target_a = y
        target_b = y[index, :]

        mixed_y = (target_a + target_b).clamp(max=1.0)

        return mixed_x, mixed_y