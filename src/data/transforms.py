import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional as F_audio
import numpy as np
from src.configs.config import cfg

class InstanceNormalize(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        return (x - mean) / (std + self.eps)

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

def _compute_deltas(mel_tensor):
    delta_tensor = F_audio.compute_deltas(mel_tensor)
    delta2_tensor = F_audio.compute_deltas(delta_tensor)
    image_tensor = torch.cat([mel_tensor, delta_tensor, delta2_tensor], dim=0)
    return image_tensor

def get_transforms(data='train'):
    if data == 'train':
        albumentations_list = [
            A.GaussNoise(
                var_limit=cfg.gaussian_noise_limit,
                mean=0, 
                per_channel=True, 
                p=cfg.gaussian_noise_prob
            ),
            ToTensorV2()
        ]

        base_pipeline = A.Compose(albumentations_list)
        norm_layer = InstanceNormalize()
        
        spec_aug = SpecAugmentation(
            time_mask_param=cfg.spec_aug_time_mask,
            freq_mask_param=cfg.spec_aug_freq_mask,
            prob=cfg.spec_aug_prob
        )

        def transform_fn(image, **kwargs):
            augmented = base_pipeline(image=image)
            mel_tensor = augmented['image'].float()
            image_tensor = _compute_deltas(mel_tensor)
            image_tensor = norm_layer(image_tensor)

            if cfg.spec_aug_time_mask > 0 or cfg.spec_aug_freq_mask > 0:
                image_tensor = spec_aug(image_tensor)
                
            return {'image': image_tensor}

        return transform_fn

    elif data == 'valid':
        base_pipeline = A.Compose([ToTensorV2()])
        norm_layer = InstanceNormalize()

        def transform_fn_val(image, **kwargs):
            augmented = base_pipeline(image=image)
            mel_tensor = augmented['image'].float()
            image_tensor = _compute_deltas(mel_tensor)
            image_tensor = norm_layer(image_tensor)
            
            return {'image': image_tensor}

        return transform_fn_val