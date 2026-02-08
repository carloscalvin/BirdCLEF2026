import numpy as np
import torch

class MixupCutmix:
    def __init__(self, mixup_prob=0.5, cutmix_prob=0.5, alpha=1.0):
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.alpha = alpha

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, batch, target):
        r = np.random.rand(1)
        if r < self.mixup_prob:
            # --- MixUp ---
            lam = np.random.beta(self.alpha, self.alpha)
            rand_index = torch.randperm(batch.size()[0]).to(batch.device)
            target_a = target
            target_b = target[rand_index]
            mixed_x = lam * batch + (1 - lam) * batch[rand_index]
            return mixed_x, target_a, target_b, lam, 'mixup'
        
        elif r < (self.mixup_prob + self.cutmix_prob):
            # --- CutMix ---
            lam = np.random.beta(self.alpha, self.alpha)
            rand_index = torch.randperm(batch.size()[0]).to(batch.device)
            target_a = target
            target_b = target[rand_index]
            
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(batch.size(), lam)
            mixed_x = batch.clone()
            mixed_x[:, :, bbx1:bbx2, bby1:bby2] = batch[rand_index, :, bbx1:bbx2, bby1:bby2]

            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch.size()[-1] * batch.size()[-2]))
            return mixed_x, target_a, target_b, lam, 'cutmix'
        
        else:
            return batch, target, target, 1.0, 'none'