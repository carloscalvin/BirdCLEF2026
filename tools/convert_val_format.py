import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.configs.config import cfg

def time_str_to_seconds(time_str: str) -> int:
    parts = time_str.strip().split(':')
    if len(parts) == 3:
        h, m, s = [int(p) for p in parts]
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = [int(p) for p in parts]
        return m * 60 + s
    else:
        return int(parts[0])

def convert_dataset(new_val_csv: str, sample_sub_csv: str, output_csv: str):
    print(f"[*] Leyendo dataset original: {new_val_csv}")
    df_raw = pd.read_csv(new_val_csv)
    
    print(f"[*] Leyendo sample submission para mapear clases: {sample_sub_csv}")
    df_sub = pd.read_csv(sample_sub_csv)
    
    classes = sorted([col for col in df_sub.columns if col != 'row_id'])
    print(f"[*] Encontradas {len(classes)} clases en el sample_submission.")

    converted_rows = []

    print("[*] Convirtiendo formato...")
    for _, row in df_raw.iterrows():
        base_filename = row['filename'].replace('.ogg', '')
        end_sec = time_str_to_seconds(row['end'])
        row_id = f"{base_filename}_{end_sec}"
        labels_in_chunk = set(str(row['primary_label']).split(';'))
        new_row = {'row_id': row_id}
        for cls in classes:
            new_row[cls] = 1.0 if cls in labels_in_chunk else 0.0
            
        converted_rows.append(new_row)

    df_converted = pd.DataFrame(converted_rows)
    final_cols = ['row_id'] + classes
    df_converted = df_converted[final_cols]

    df_converted.to_csv(output_csv, index=False)
    print(f"[*] ¡Éxito! Dataset convertido y guardado en: {output_csv}")
    print(f"[*] Filas procesadas: {len(df_converted)}")

if __name__ == "__main__":

    RAW_VAL_PATH = os.path.join(cfg.dataset_root, "train_soundscapes_labels.csv")
    SAMPLE_SUB_PATH = os.path.join(cfg.dataset_root, "sample_submission.csv")
    OUTPUT_PATH = os.path.join(cfg.dataset_root, "val_soundscape.csv")

    convert_dataset(RAW_VAL_PATH, SAMPLE_SUB_PATH, OUTPUT_PATH)