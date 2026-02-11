import os
import cv2
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.configs.config import cfg

class BirdDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, mode='train', class_names=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.class_names = class_names
        self.use_cache = cfg.use_ram_cache and (mode == 'train')
        self.audio_cache = {}
        
        if self.class_names:
            self.class_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
            self.num_classes = len(self.class_names)

        if self.mode == 'train':
            self.df = self.df[self.df['primary_label'].isin(self.class_names)].reset_index(drop=True)
            self.file_to_label = self.df.set_index('filename')['primary_label'].to_dict()
            self.file_paths = self.df['filename'].tolist()
            print(f"[TRAIN] Dataset 'On-the-Fly' cargado. {len(self.file_paths)} archivos de audio.")
            if self.use_cache:
                self.cache_all_audio()
        else:
            print(f"[VALID] Dataset pre-procesado cargado. {len(self.df)} muestras de soundscape.")

    def __len__(self):
        if self.mode == 'train':
            return len(self.file_paths)
        else:
            return len(self.df)

    def compute_melspec(self, y):
        melspec = librosa.feature.melspectrogram(
            y=y, sr=cfg.sr, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax,
            n_fft=cfg.n_fft, hop_length=cfg.hop_length
        )
        melspec = librosa.power_to_db(melspec, ref=np.max)
        melspec = melspec - melspec.min()
        melspec = melspec / (melspec.max() + 1e-6)
        melspec = (melspec * 255).astype(np.uint8)
        melspec = np.flip(melspec, axis=0)
        return melspec

    def cache_all_audio(self):
        def load_single_file(filename):
            path = os.path.join(self.root_dir, filename)
            y, _ = librosa.load(path, sr=cfg.sr)
            return filename, y.astype(np.float32)
        with ThreadPoolExecutor(max_workers=cfg.num_cache_workers) as executor:
            results = list(tqdm(executor.map(load_single_file, self.file_paths), total=len(self.file_paths), unit="files"))
        for filename, audio_data in results:
            self.audio_cache[filename] = audio_data
        print(f"[CACHE] Carga completada. {len(self.audio_cache)} archivos en memoria.\n")

    def load_and_crop_audio(self, filename):
        if self.use_cache:
            y = self.audio_cache[filename]
        else:
            file_path = os.path.join(self.root_dir, filename)
            try:
                y, _ = librosa.load(file_path, sr=cfg.sr)
                y = y.astype(np.float32)
            except Exception as e:
                print(f"Error cargando {file_path}: {e}")
                raise

        target_len = cfg.sr * cfg.duration
        
        if len(y) < target_len:
            padding = target_len - len(y)
            offset = padding // 2
            y = np.pad(y, (offset, target_len - len(y) - offset), 'constant')
        elif len(y) > target_len:
            start = np.random.randint(0, len(y) - target_len)
            y = y[start : start + target_len]

        return y.astype(np.float32)

    def __getitem__(self, idx):
        if self.mode == 'train':
            filename = self.file_paths[idx]
            label_str = self.file_to_label[filename]
            y = self.load_and_crop_audio(filename)
            image_gray = self.compute_melspec(y)
            image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
            target = np.zeros(self.num_classes, dtype=np.float32)
            if label_str in self.class_to_idx:
                cls_idx = self.class_to_idx[label_str]
                target[cls_idx] = 1.0
            
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