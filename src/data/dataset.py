import os
import cv2
import torch
import numpy as np
import ast
from torch.utils.data import Dataset

class BirdDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, mode='train', class_names=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        self.class_names = class_names
        self.class_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        if self.mode == 'train':
            df = df[df['primary_label'].isin(self.class_names)].reset_index(drop=True)
            self.file_to_chunks = df.groupby('filename')['chunk_name'].apply(list).to_dict()
            self.file_names = list(self.file_to_chunks.keys())
            self.file_to_label = df.drop_duplicates('filename').set_index('filename')['primary_label'].to_dict()
            self.file_to_secondary = {}
            temp_df = df.drop_duplicates('filename').set_index('filename')
            for fname, row in temp_df.iterrows():
                labels = ast.literal_eval(row.get('secondary_labels'))
                self.file_to_secondary[fname] = [l for l in labels if l]
            print(f"[TRAIN] Dataset cargado. {len(self.file_names)} archivos únicos. {self.num_classes} clases.")

        else:
            self.df = df
            print(f"[VALID] Dataset cargado. {len(self.df)} muestras de soundscape.")

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

            target = np.zeros(self.num_classes, dtype=np.float32)
            cls_idx = self.class_to_idx[label_str]
            target[cls_idx] = 1.0

            sec_labels = self.file_to_secondary.get(orig_filename)
            for s_label in sec_labels:
                s_idx = self.class_to_idx[s_label]
                target[s_idx] = 1.0

        else:
            row = self.df.iloc[idx]
            chunk_name = row['chunk_name']
            target = row['targets']
            if not isinstance(target, np.ndarray):
                target = np.array(target)
            target = target.astype(np.float32)

        file_path = os.path.join(self.root_dir, chunk_name)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = image.astype(np.float32) / 255.0
            image = torch.tensor(image).permute(2, 0, 1)

        return image, torch.tensor(target, dtype=torch.float32)