import os
import torch
import numpy as np
import ast
from torch.utils.data import Dataset

class BirdDataset(Dataset):
    def __init__(
        self,
        df,
        root_dir,
        transform,
        mode='train',
        class_names=None,
        pseudo_threshold=None,
        pseudo_use_soft_labels=False,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

        self.class_names = class_names
        self.pseudo_threshold = pseudo_threshold
        self.pseudo_use_soft_labels = bool(pseudo_use_soft_labels)
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
        elif self.mode == 'pseudo':
            self.df = df
            print(f"[PSEUDO] Dataset cargado. {len(self.df)} muestras confiables (hard labels).")
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
        elif self.mode == 'pseudo':
            row = self.df.iloc[idx]
            chunk_name = row['chunk_name']
            target = row['targets']
            if not isinstance(target, np.ndarray):
                target = np.array(target)
            target = target.astype(np.float32)
            if self.pseudo_use_soft_labels:
                # Soft labels en clases de confianza (>= threshold), cero en el resto.
                # Esto evita inyectar ruido de clases con prob baja en el mixup
                # pero preserva la incertidumbre del teacher en las clases activas.
                mask = target >= self.pseudo_threshold
                target = np.where(mask, target, 0.0).astype(np.float32)
            else:
                target = (target >= self.pseudo_threshold).astype(np.float32)
        else:
            row = self.df.iloc[idx]
            chunk_name = row['chunk_name']
            target = row['targets']
            if not isinstance(target, np.ndarray):
                target = np.array(target)
            target = target.astype(np.float32)

        file_path = os.path.join(self.root_dir, chunk_name)
        spec_arr = np.load(file_path)
        spec_arr = spec_arr.astype(np.float32)
        spec_arr = spec_arr[:, :, np.newaxis]

        augmented = self.transform(image=spec_arr)
        image = augmented['image']

        return image, torch.tensor(target, dtype=torch.float32)