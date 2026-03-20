import os
import sys
import ast
import numpy as np
import pandas as pd
import torch
import collections
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.configs.config import cfg
from src.models.model import BirdModel
from src.data.transforms import get_transforms

class TrainChunkDataset(Dataset):
    def __init__(self, df, root_dir):
        self.df = df
        self.root_dir = root_dir
        self.transform = get_transforms('valid')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        chunk_name = row['chunk_name']
        
        file_path = os.path.join(self.root_dir, chunk_name)
        spec_arr = np.load(file_path)
        spec_arr = spec_arr.astype(np.float32)
        spec_arr = spec_arr[:, :, np.newaxis]

        augmented = self.transform(image=spec_arr)
        image = augmented['image']

        return image, idx

def run_enrichment(weights_path, threshold=0.85):
    print("\n--- Iniciando enriquecimiento de etiquetas (hard pseudo-labeling) ---")
    
    class_names = pd.read_csv(cfg.classes_order_path, header=None)[0].tolist()
    num_classes = len(class_names)

    input_csv = os.path.join(cfg.data_dir, "train_processed.csv")
    output_csv = os.path.join(cfg.data_dir, "train_enriched.csv")
    print(f"[*] Leyendo dataset base: {input_csv}")
    
    df = pd.read_csv(input_csv)
    if cfg.fast_dev_run:
        df = df.head(100)
        print("[!] Modo FAST RUN: Procesando solo 100 chunks.")

    ds = TrainChunkDataset(df, cfg.preprocess_train_dir)
    loader = DataLoader(ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BirdModel(model_name=cfg.model_cfg.model_name, num_classes=num_classes, pretrained=False)
    
    print(f"[*] Cargando pesos desde: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = np.zeros((len(df), num_classes), dtype=np.float32)
    
    print("[*] Evaluando chunks de entrenamiento...")
    with torch.no_grad():
        for images, indices in tqdm(loader, leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds[indices.numpy()] = probs

    np.save(os.path.join(cfg.output_dir, "train_raw_preds.npy"), all_preds)

    print(f"[*] Inyectando predicciones > {threshold} como etiquetas duras...")
    
    preds_boolean = all_preds > threshold
    
    total_new_labels = 0
    chunks_modified = 0
    new_labels_counter = collections.Counter()
    
    for idx in tqdm(range(len(df)), leave=False):
        row = df.iloc[idx]
        primary = row['primary_label']
        sec_str = row['secondary_labels']
        secs = ast.literal_eval(sec_str)


        class_detected_idx = np.where(preds_boolean[idx])[0]
        
        new_secs = set(secs)
        added_in_chunk = False
        
        for c_idx in class_detected_idx:
            class_name = class_names[c_idx]
            if class_name != primary and class_name not in secs:
                new_secs.add(class_name)
                new_labels_counter[class_name] += 1
                total_new_labels += 1
                added_in_chunk = True

        if added_in_chunk:
            chunks_modified += 1

        df.at[idx, 'secondary_labels'] = str(list(new_secs))

    df.to_csv(output_csv, index=False)

    print("\n==================================================")
    print(f"Umbral utilizado: {threshold}")
    print(f"Chunks totales procesados    : {len(df)}")
    print(f"Chunks modificados           : {chunks_modified} ({(chunks_modified/len(df))*100:.2f}%)")
    print(f"Nuevas etiquetas inyectadas  : {total_new_labels}")
    print("\nTop 10 especies secundarias más inyectadas:")
    for especie, count in new_labels_counter.most_common(10):
        print(f"  - {especie}: {count} veces")
    print("==================================================\n")
    
    print(f"[*] ¡Éxito! Dataset enriquecido guardado en: {output_csv}")

if __name__ == "__main__":
    weights_path = os.path.join(cfg.output_dir, f"{cfg.model_cfg.model_name}_to_enrich_labels.pth")
    run_enrichment(weights_path, cfg.train_enrichment_threshold)