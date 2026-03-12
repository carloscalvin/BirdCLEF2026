import pandas as pd

def merge_val_soundscapes():

    path_2025 = "dataset/birdclef-2025/val_soundscape.csv"
    path_2026 = "dataset/birdclef-2026/val_soundscape.csv"
    out_path  = "dataset/birdclef-2026/val_soundscape_combined.csv"

    print(f"[*] Leyendo CSV 2025: {path_2025}")
    df_2025 = pd.read_csv(path_2025)
    
    print(f"[*] Leyendo CSV 2026: {path_2026}")
    df_2026 = pd.read_csv(path_2026)

    classes_2026 = [c for c in df_2026.columns if c != "row_id"]

    cols_2025 = set(df_2025.columns) - {"row_id"}
    cols_to_drop = [c for c in cols_2025 if c not in classes_2026]
    cols_to_add = [c for c in classes_2026 if c not in cols_2025]

    print(f"[*] Especies a eliminar de 2025: {len(cols_to_drop)}")
    print(f"[*] Especies a añadir a 2025 (como 0.0): {len(cols_to_add)}")

    df_2025 = df_2025.drop(columns=cols_to_drop)
    
    for c in cols_to_add:
        df_2025[c] = 0.0

    df_2025 = df_2025[['row_id'] + classes_2026]

    df_combined = pd.concat([df_2026, df_2025], ignore_index=True)
    
    df_combined.to_csv(out_path, index=False)
    print(f"\n[+] Guardado CSV combinado con {len(df_combined)} filas en: {out_path}")

if __name__ == "__main__":
    merge_val_soundscapes()