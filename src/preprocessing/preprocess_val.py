import os
import sys
import pandas as pd
import numpy as np
import cv2
import librosa
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.configs.config import cfg
from src.preprocessing.preprocess import compute_melspec

def preprocess_validation():
    print(f"Leyendo teacher CSV: {cfg.teacher_preds_path}")
    df = pd.read_csv(cfg.teacher_preds_path)

    non_class_cols = ['row_id']
    class_cols = [c for c in df.columns if c not in non_class_cols]
    class_cols = sorted(class_cols)

    print(f"Detectadas {len(class_cols)} clases en validación.")

    pd.Series(class_cols).to_csv(cfg.classes_order_path, index=False, header=False)

    valid_data = []
    
    print(f"Procesando {len(df)} segmentos de soundscapes...")
    
    os.makedirs(cfg.preprocess_val_dir, exist_ok=True)

    missing_files = set()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        row_id = row['row_id']

        try:
            parts = row_id.rsplit('_', 1)
            base_filename = parts[0]
            end_seconds = int(parts[1])
            start_seconds = end_seconds
            audio_filename = base_filename + ".ogg"
            audio_path = os.path.join(cfg.train_soundscapes_dir, audio_filename)
            
        except ValueError:
            print(f"Error parseando row_id: {row_id}")
            continue

        if not os.path.exists(audio_path):
            if audio_filename not in missing_files:
                print(f"[Warning] Audio no encontrado: {audio_path}")
                missing_files.add(audio_filename)
            continue

        try:
            y, _ = librosa.load(audio_path, sr=cfg.sr, offset=start_seconds, duration=cfg.duration)
            spec = compute_melspec(y, cfg.sr)

            save_name = row_id + ".png"
            save_path = os.path.join(cfg.preprocess_val_dir, save_name)
            cv2.imwrite(save_path, spec)

            soft_targets = row[class_cols].values.astype(np.float32)
            
            valid_data.append({
                'chunk_name': save_name,
                'targets': soft_targets,
                'row_id': row_id
            })
            
        except Exception as e:
            print(f"Error procesando {row_id} en {audio_path}: {e}")

    df_val_processed = pd.DataFrame(valid_data)
    df_val_processed.to_pickle(cfg.val_processed_path)
    
    print(f"Procesado completado. Guardado en {cfg.val_processed_path}")
    print(f"Total muestras validación generadas: {len(df_val_processed)}")

if __name__ == "__main__":
    preprocess_validation()