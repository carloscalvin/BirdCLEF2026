import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(data='train'):
    if data == 'train':
        return A.Compose([
            # A.CoarseDropout(
            #    max_holes=8,
            #    max_height=16,
            #    max_width=16,
            #    min_holes=2,
            #    fill_value=0,
            #    p=0.5
            # ),
            # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            # A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2(),
        ])

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