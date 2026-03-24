import os
import sys
import glob
from collections import Counter
import soundfile as sf
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.configs.config import cfg

def analyze_directory_sr(directory, name):
    print(f"\n[*] Escaneando archivos en: {name}")
    if not os.path.exists(directory):
        print(f"[!] La ruta no existe: {directory}")
        return

    search_pattern = os.path.join(directory, "**", "*.ogg")
    audio_files = glob.glob(search_pattern, recursive=True)
    
    if not audio_files:
        print(f"[!] No se encontraron archivos .ogg en {directory}")
        return

    print(f"[*] Encontrados {len(audio_files)} archivos. Leyendo metadatos...")
    
    sr_counter = Counter()
    
    for file_path in tqdm(audio_files, desc=f"Analizando {name}"):
        try:
            info = sf.info(file_path)
            sr_counter[info.samplerate] += 1
        except Exception as e:
            print(f"Error leyendo {os.path.basename(file_path)}: {e}")

    print(f"\n=== RESULTADOS PARA {name.upper()} ===")
    total_files = sum(sr_counter.values())
    for sr, count in sr_counter.most_common():
        porcentaje = (count / total_files) * 100
        print(f" -> {sr} Hz: {count} archivos ({porcentaje:.2f}%)")

def main():
    print("Iniciando análisis de sample rates (frecuencias de muestreo)...\n")

    analyze_directory_sr(cfg.train_audio_dir, "train audio (especies)")
    analyze_directory_sr(cfg.train_soundscapes_dir, "train soundscapes")
    analyze_directory_sr(cfg.test_soundscapes_dir, "test soundscapes")

    print("\n[+] Análisis completado.")

if __name__ == "__main__":
    main()
