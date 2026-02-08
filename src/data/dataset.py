import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class BirdDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.unique_labels = sorted(df['primary_label'].unique())
        self.class_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}

        if self.mode == 'train':
            self.file_to_chunks = df.groupby('filename')['chunk_name'].apply(list).to_dict()
            self.file_names = list(self.file_to_chunks.keys())
            self.file_to_label = df.drop_duplicates('filename').set_index('filename')['primary_label'].to_dict()
            print(f"[{mode.upper()}] Dataset agrupado por Archivo. Length: {len(self.file_names)}")
        else:
            self.df = df
            print(f"[{mode.upper()}] Dataset plano (todos los chunks). Length: {len(self.df)}")

    def __len__(self):
        if self.mode == 'train':
            return len(self.file_names)
        else:
            return len(self.df)

    def __getitem__(self, idx):
        if self.mode == 'train':
            orig_filename = self.file_names[idx]
            chunk_list = self.file_to_chunks[orig_filename]
            chunk_name = np.random.choice(chunk_list)
            label_str = self.file_to_label[orig_filename]
        else:
            row = self.df.iloc[idx]
            chunk_name = row['chunk_name']
            label_str = row['primary_label']

        file_path = os.path.join(self.root_dir, chunk_name)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = image.astype(np.float32) / 255.0
            image = torch.tensor(image).permute(2, 0, 1)

        label = self.class_to_idx[label_str]
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label