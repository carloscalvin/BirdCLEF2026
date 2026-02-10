import os
import sys
import glob
import math
import numpy as np
import pandas as pd
import torch
import librosa
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.configs.config import cfg
from src.models.model import BirdModel

class TestDataset(Dataset):
    def __init__(self, audio_files, sr=32000, duration=5):
        self.audio_files = audio_files
        self.sr = sr
        self.duration = duration
        self.data = []

        for file_path in self.audio_files:
            filename = os.path.basename(file_path)
            base_name = filename.split('.')[0]
            try:
                audio_len_sec = librosa.get_duration(path=file_path)
                chunks = math.ceil(audio_len_sec / duration)
                
                for i in range(chunks):
                    end_sec = (i + 1) * duration
                    row_id = f"{base_name}_{end_sec}"
                    
                    self.data.append({
                        'file_path': file_path,
                        'start': i * duration,
                        'end': end_sec,
                        'row_id': row_id
                    })
            except Exception as e:
                print(f"Error leyendo {filename}: {e}")

    def __len__(self):
        return len(self.data)

    def compute_melspec(self, y):
        melspec = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax,
            n_fft=cfg.n_fft, hop_length=cfg.hop_length
        )
        melspec = librosa.power_to_db(melspec, ref=np.max)
        melspec = melspec - melspec.min()
        melspec = melspec / (melspec.max() + 1e-6)
        melspec = (melspec * 255).astype(np.uint8)
        melspec = np.flip(melspec, axis=0)
        return melspec

    def __getitem__(self, idx):
        item = self.data[idx]
        y, _ = librosa.load(
            item['file_path'], 
            sr=self.sr, 
            offset=item['start'], 
            duration=self.duration
        )

        target_len = self.sr * self.duration
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        spec = self.compute_melspec(y)
        img = cv2.cvtColor(spec, cv2.COLOR_GRAY2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = torch.tensor(img).permute(2, 0, 1).float()
        
        return img, item['row_id']

def run_inference(weights_path, clasess_path, files_path):
    print(f"[*] Buscando archivos en: {files_path}")
    audio_files = glob.glob(os.path.join(files_path, "*.ogg"))

    if not os.path.exists(clasess_path):
        raise FileNotFoundError(f"No se encuentra classes_order.csv en {clasess_path}")
    
    class_names = pd.read_csv(clasess_path, header=None)[0].tolist()
    num_classes = len(class_names)
    print(f"[*] Clases detectadas: {num_classes}")

    ds = TestDataset(audio_files, sr=cfg.sr, duration=cfg.duration)
    loader = DataLoader(ds, batch_size=cfg.batch_size*2, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BirdModel(
        model_name=cfg.model_cfg.model_name, 
        num_classes=num_classes, 
        pretrained=False
    )
    
    print(f"[*] Cargando pesos desde: {weights_path}")
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"[ERROR] Fallo cargando pesos: {e}")
        return

    model.to(device)
    model.eval()

    all_probs = []
    all_row_ids = []

    print("[*] Iniciando inferencia...")
    with torch.no_grad():
        for images, row_ids in tqdm(loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_probs.append(probs)
            all_row_ids.extend(row_ids)

    all_probs = np.concatenate(all_probs)

    df_sub = pd.DataFrame(all_probs, columns=class_names)
    df_sub.insert(0, "row_id", all_row_ids)
    df_sub.to_csv("submission.csv", index=False)
    print(f"[*] Submission guardado: submission.csv con {len(df_sub)} filas.")

if __name__ == "__main__":
    weights_path = os.path.join(cfg.output_dir, f"{cfg.model_cfg.model_name}_best_teacher.pth")
    classes_path = cfg.classes_order_path
    files_path = cfg.test_soundscapes_dir
    run_inference(weights_path, classes_path, files_path)