import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
from src.configs.config import cfg

class SpecAugmentation(nn.Module):
    def __init__(self, time_mask_param, freq_mask_param, prob=0.5):
        super().__init__()
        self.prob = prob
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param) if time_mask_param > 0 else None
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param) if freq_mask_param > 0 else None

    def forward(self, x):
        if np.random.rand() > self.prob:
            return x

        out = x.clone()

        if self.time_mask:
            out = self.time_mask(out)
        if self.freq_mask:
            out = self.freq_mask(out)
            
        return out

def get_transforms(data='train'):
    if data == 'train':
        transforms_list = [
            A.GaussNoise(
                var_limit=cfg.gaussian_noise_limit, 
                mean=0, 
                per_channel=True, 
                p=cfg.gaussian_noise_prob
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()
        ]

        base_pipeline = A.Compose(transforms_list)

        spec_aug = SpecAugmentation(
            time_mask_param=cfg.spec_aug_time_mask,
            freq_mask_param=cfg.spec_aug_freq_mask,
            prob=cfg.spec_aug_prob
        )

        def transform_fn(image, **kwargs):
            augmented = base_pipeline(image=image)
            image_tensor = augmented['image']

            if cfg.spec_aug_time_mask > 0 or cfg.spec_aug_freq_mask > 0:
                image_tensor = spec_aug(image_tensor)
                
            return {'image': image_tensor}

        return transform_fn

    elif data == 'valid':
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2(),
        ])