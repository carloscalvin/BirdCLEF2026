import os
import sys
import pandas as pd
import numpy as np
import librosa
import cv2
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.configs.config import cfg

def compute_melspec(y, sr):
    melspec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax,
        n_fft=cfg.n_fft, hop_length=cfg.hop_length
    )
    melspec = librosa.power_to_db(melspec, ref=np.max)
    melspec = melspec - melspec.min()
    melspec = melspec / (melspec.max() + 1e-6)
    melspec = (melspec * 255).astype(np.uint8)
    melspec = np.flip(melspec, axis=0)
    return melspec

def process_audio_file(row):
    file_path = os.path.join(cfg.train_audio_dir, row['filename'])
    label = row['primary_label']
    filename_base = row['filename'].replace('/', '_').replace('.ogg', '')
    new_rows = []
    try:
        y, sr = librosa.load(file_path, sr=cfg.sr)
        samples_per_chunk = cfg.sr * cfg.duration
        total_samples = len(y)
        if total_samples < samples_per_chunk:
            y = np.pad(y, (0, samples_per_chunk - total_samples))
            chunks = [y]
        else:
            num_chunks = int(np.ceil(total_samples / samples_per_chunk))
            chunks = []
            for i in range(num_chunks):
                start = i * samples_per_chunk
                end = start + samples_per_chunk
                chunk = y[start:end]
                if len(chunk) < samples_per_chunk:
                    chunk = np.pad(chunk, (0, samples_per_chunk - len(chunk)))
                chunks.append(chunk)
        for i, chunk in enumerate(chunks):
            spec_img = compute_melspec(chunk, cfg.sr)
            save_name = f"{filename_base}_chunk{i}.png"
            save_path = os.path.join(cfg.preprocess_train_dir, save_name)
            cv2.imwrite(save_path, spec_img)
            new_rows.append({
                'filename': row['filename'],
                'chunk_name': save_name,
                'primary_label': label,
                'chunk_index': i
            })
    except Exception as e:
        print(f"Error procesando {file_path}: {e}")
        return []
    return new_rows

def run_preprocessing():
    print(f"Leyendo {cfg.train_csv_path}...")
    df = pd.read_csv(cfg.train_csv_path)
    if cfg.fast_dev_run:
        df = df.head(50)
        print("Modo FAST RUN: Procesando solo 50 archivos.")

    print(f"Procesando {len(df)} archivos de audio. Generando chunks de {cfg.duration}s...")
    
    results = Parallel(n_jobs=-1)(
        delayed(process_audio_file)(row) 
        for _, row in tqdm(df.iterrows(), total=len(df))
    )
    
    all_chunks = [item for sublist in results for item in sublist]

    processed_df = pd.DataFrame(all_chunks)
    processed_csv_path = os.path.join(cfg.data_dir, "train_processed.csv")
    processed_df.to_csv(processed_csv_path, index=False)
    
    print("\n¡Preprocesamiento completado!")
    print(f"Total chunks generados: {len(processed_df)}")
    print(f"Imágenes guardadas en: {cfg.preprocess_train_dir}")
    print(f"Nuevo CSV guardado en: {processed_csv_path}")

if __name__ == "__main__":
    run_preprocessing()