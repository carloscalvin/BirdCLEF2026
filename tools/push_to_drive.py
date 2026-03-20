import os
import shutil
import zipfile
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_NAME = "BirdCLEF2026_Codebase"
DRIVE_DESTINATION_DIR = None 

INCLUDE_DIRS = ["src"]
INCLUDE_FILES = [
    "outputs/tf_efficientnet_b0_ns_to_enrich_labels.pth",
    "requirements.txt"
]

def zip_project():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"{OUTPUT_NAME}_{timestamp}.zip"
    output_path = os.path.join(PROJECT_ROOT, zip_filename)

    print(f"[*] Empaquetando proyecto en {output_path}...")

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder in INCLUDE_DIRS:
            folder_path = os.path.join(PROJECT_ROOT, folder)
            if not os.path.exists(folder_path):
                print(f"[!] Warning: Carpeta no encontrada {folder}")
                continue

            for root, _, files in os.walk(folder_path):
                if '__pycache__' in root: continue

                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, PROJECT_ROOT)
                    zipf.write(file_path, arcname)

        for file in INCLUDE_FILES:
            file_path = os.path.join(PROJECT_ROOT, file)
            if os.path.exists(file_path):
                zipf.write(file_path, file)

    print(f"[*] Zip creado: {os.path.basename(output_path)}")
    
    if DRIVE_DESTINATION_DIR and os.path.exists(DRIVE_DESTINATION_DIR):
        print(f"[*] Moviendo a Google Drive: {DRIVE_DESTINATION_DIR}")
        shutil.move(output_path, os.path.join(DRIVE_DESTINATION_DIR, zip_filename))
        print("[*] ¡Subida completada!")
    else:
        print(f"[*] Sube este archivo manualmente a tu Google Drive si no configuraste la ruta automática.")

if __name__ == "__main__":
    zip_project()