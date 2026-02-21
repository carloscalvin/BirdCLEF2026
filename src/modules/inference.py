import os
import sys
import glob
import math
import numpy as np
import pandas as pd
import torch
import librosa
import noisereduce as nr
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
from src.data.transforms import get_transforms
from src.preprocessing.preprocess import compute_melspec
from src.modules.postprocess import PostProcessor

class TestDataset(Dataset):
    def __init__(self, audio_files, n_fft, hop_length, sr=32000, duration=5):
        self.audio_files = audio_files
        self.sr = sr
        self.duration = duration
        self.transform = get_transforms('valid')
        self.data = []
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.use_sliding = cfg.use_sliding
        self.overlaps = cfg.overlaps

        for file_path in self.audio_files:
            filename = os.path.basename(file_path)
            base_name = filename.split('.')[0]
            audio_len_sec = librosa.get_duration(path=file_path)
            chunks = math.ceil(audio_len_sec / duration)
            
            for i in range(chunks):
                target_end = (i + 1) * duration
                target_start = i * duration
                
                if not self.use_sliding:
                    row_id = f"{base_name}_{target_end}"
                    self.data.append({
                        'file_path': file_path,
                        'start': target_start,
                        'end': target_end,
                        'row_id': row_id
                    })
                else:
                    half_window = duration / 2.0
                    shifts = np.linspace(-half_window/2, half_window/2, self.overlaps)
                    
                    for shift in shifts:
                        win_start = max(0.0, target_start + shift)
                        row_id_temp = f"{base_name}_{target_end}|{win_start:.2f}"
                        self.data.append({
                            'file_path': file_path,
                            'start': win_start,
                            'end': win_start + duration,
                            'row_id': row_id_temp
                        })

    def __reduce_noise__(self, signal):
        if not cfg.reduce_noise:
            return signal

        y = nr.reduce_noise(y=signal,
                            sr=self.sr,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            prop_decrease=cfg.reduce_noise_prop_decrease,
                            stationary=cfg.reduce_noise_stationary)
        return y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        y_orig, _ = librosa.load(
            item['file_path'], 
            sr=self.sr, 
            offset=item['start'], 
            duration=self.duration
        )

        y = self.__reduce_noise__(y_orig)

        target_len = self.sr * self.duration
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))

        spec_arr = compute_melspec(y, cfg.sr)
        spec_arr = spec_arr.astype(np.float32)
        spec_arr = spec_arr[:, :, np.newaxis]
        augmented = self.transform(image=spec_arr)
        image = augmented['image']

        return image, item['row_id']

def run_inference(weights_path, clasess_path, files_path):
    print(f"[*] Buscando archivos en: {files_path}")
    audio_files = glob.glob(os.path.join(files_path, "*.ogg"))

    if not os.path.exists(clasess_path):
        raise FileNotFoundError(f"No se encuentra classes_order.csv en {clasess_path}")
    
    class_names = pd.read_csv(clasess_path, header=None)[0].tolist()
    num_classes = len(class_names)
    print(f"[*] Clases detectadas: {num_classes}")

    ds = TestDataset(audio_files, cfg.n_fft, cfg.hop_length, sr=cfg.sr, duration=cfg.duration)
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
    pp = PostProcessor(cfg)
    all_probs, all_row_ids = pp.run(all_probs, all_row_ids)
    df_sub = pd.DataFrame(all_probs, columns=class_names)
    df_sub.insert(0, "row_id", all_row_ids)
    df_sub.to_csv("submission.csv", index=False)
    print(f"[*] Submission guardado: submission.csv con {len(df_sub)} filas.")

if __name__ == "__main__":
    weights_path = os.path.join(cfg.output_dir, f"{cfg.model_cfg.model_name}_best_teacher.pth")
    classes_path = cfg.classes_order_path
    files_path = cfg.test_soundscapes_dir
    run_inference(weights_path, classes_path, files_path)