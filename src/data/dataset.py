import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class BirdDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.unique_labels = sorted(df['primary_label'].unique())
        self.class_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['chunk_name']
        file_path = os.path.join(self.root_dir, file_name)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = image.astype(np.float32) / 255.0
            image = torch.tensor(image).permute(2, 0, 1)

        label_str = row['primary_label']
        label = self.class_to_idx[label_str]
        
        return image, torch.tensor(label, dtype=torch.long)