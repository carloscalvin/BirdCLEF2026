import os
import json
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_SLUG = "birdclef2026-code"
DATASET_TITLE = "BirdCLEF 2026 Codebase"
TEMP_DIR = os.path.join(PROJECT_ROOT, "kaggle_dataset_temp")

INCLUDE_DIRS = ["src"]
INCLUDE_FILES = [
    "outputs/regnety_008.pycls_in1k_best_teacher.pth",
    "dataset/birdclef-2026/classes_order.csv"
]

def prepare_dataset():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    print(f"Copiando archivos a {TEMP_DIR}...")

    for d in INCLUDE_DIRS:
        src = os.path.join(PROJECT_ROOT, d)
        dst = os.path.join(TEMP_DIR, d)
        if os.path.exists(src):
            shutil.copytree(src, dst)
        else:
            print(f"[Warning] Carpeta no encontrada: {src}")

    for f in INCLUDE_FILES:
        src = os.path.join(PROJECT_ROOT, f)
        dst = os.path.join(TEMP_DIR, os.path.basename(f))
        dst_full = os.path.join(TEMP_DIR, f)
        os.makedirs(os.path.dirname(dst_full), exist_ok=True)
        
        if os.path.exists(src):
            shutil.copy2(src, dst_full)
        else:
            print(f"[Warning] Archivo no encontrado: {src}")

    metadata = {
        "title": DATASET_TITLE,
        "id": f"{api.get_config_value('username')}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}]
    }
    
    with open(os.path.join(TEMP_DIR, "dataset-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

def push_to_kaggle():
    api = KaggleApi()
    api.authenticate()
    
    user_name = api.get_config_value('username')
    dataset_id = f"{user_name}/{DATASET_SLUG}"
    
    print(f"Checkeando si el dataset {dataset_id} existe...")
    try:
        api.dataset_list_files(dataset_id)
        exists = True
    except:
        exists = False

    if exists:
        print("El dataset existe. Creando nueva versión...")
        api.dataset_create_version(
            TEMP_DIR, 
            version_notes="Auto-update from script",
            dir_mode="zip"
        )
    else:
        print("El dataset no existe. Creándolo...")
        api.dataset_create_new(
            TEMP_DIR,
            dir_mode="zip"
        )
    
    print("¡Subida completada!")

if __name__ == "__main__":
    api = KaggleApi()
    api.authenticate()
    
    prepare_dataset()
    push_to_kaggle()