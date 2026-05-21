import os
import sys
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.configs.config import cfg
from src.preprocessing.preprocess import compute_melspec

def preprocess_soundscapes(preds_path, preprocess_dir, processed_path, pseudo=False):
    print(f"Leyendo teacher CSV: {preds_path}")
    df = pd.read_csv(preds_path)

    non_class_cols = ['row_id']
    class_cols = [c for c in df.columns if c not in non_class_cols]
    class_cols = sorted(class_cols)

    load_duration = getattr(cfg, 'train_duration', cfg.duration) if pseudo else cfg.duration

    mode_label = "PSEUDO" if pseudo else "VALIDACIÓN"
    print(f"Detectadas {len(class_cols)} clases. Modo {mode_label} — duración de carga: {load_duration} s.")

    pd.Series(class_cols).to_csv(cfg.classes_order_path, index=False, header=False)

    if pseudo:
        thresh = cfg.pseudo_threshold
        mask = (df[class_cols] >= thresh).any(axis=1)
        total_before = len(df)
        df = df[mask].reset_index(drop=True)
        kept = len(df)
        print(f"Modo pseudo: filtradas {total_before - kept} filas; quedan {kept} con pred >= {thresh}")

    valid_data = []
    
    print(f"Procesando {len(df)} segmentos de soundscapes...")
    
    os.makedirs(preprocess_dir, exist_ok=True)

    missing_files = set()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        row_id = row['row_id']

        try:
            parts = row_id.rsplit('_', 1)
            base_filename = parts[0]
            end_seconds = int(parts[1])
            start_seconds = max(0, end_seconds - cfg.duration)
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
            y, _ = librosa.load(audio_path, sr=cfg.sr, offset=start_seconds, duration=load_duration)
            
            expected_samples = int(cfg.sr * load_duration)
            if len(y) < expected_samples:
                y = np.pad(y, (0, expected_samples - len(y)), mode='constant')
            spec_arr = compute_melspec(y, cfg.sr)

            save_name = row_id + ".npy"
            save_path = os.path.join(preprocess_dir, save_name)
            np.save(save_path, spec_arr)

            soft_targets = row[class_cols].values.astype(np.float32)
            
            valid_data.append({
                'chunk_name': save_name,
                'targets': soft_targets,
                'row_id': row_id
            })
            
        except Exception as e:
            print(f"Error procesando {row_id} en {audio_path}: {e}")

    df_val_processed = pd.DataFrame(valid_data)
    df_val_processed.to_pickle(processed_path)
    
    print(f"Procesado completado. Guardado en {processed_path}")
    print(f"Total muestras validación generadas: {len(df_val_processed)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validation", "pseudo"], default="validation",
                        help="Modo: generar validación o pseudo-labels")
    args = parser.parse_args()

    if args.mode == "validation":
        print("\n=== Generando espectogramas de VALIDACIÓN ===")
        preprocess_soundscapes(cfg.teacher_preds_path, cfg.preprocess_val_dir, cfg.val_processed_path)

    elif args.mode == "pseudo":
        print("\n=== Generando espectogramas de PSEUDO-LABELS ===")
        preprocess_soundscapes(cfg.pseudo_soundscape_labels_path, cfg.preprocess_pseudo_dir, cfg.pseudo_processed_path, True)
