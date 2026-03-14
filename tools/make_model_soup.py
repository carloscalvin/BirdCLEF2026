import os
import sys
import torch
import glob
from collections import OrderedDict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.configs.config import cfg

def make_model_soup(model_dir: str, model_prefix: str, output_path: str):
    print(f"[*] Buscando checkpoints para: {model_prefix} en {model_dir}")
    
    search_pattern = os.path.join(model_dir, f"{model_prefix}_epoch_*.pth")
    checkpoints = sorted(glob.glob(search_pattern))
    
    if not checkpoints:
        print("[!] No se encontraron checkpoints. Verifica la ruta y el prefijo.")
        return

    print(f"[*] Encontrados {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"    - {os.path.basename(ckpt)}")

    print("\n[*] Iniciando model soup (uniforme)...")
    base_state_dict = torch.load(checkpoints[0], map_location='cpu', weights_only=True)

    if isinstance(base_state_dict, dict) and 'model_state_dict' in base_state_dict:
        base_state_dict = base_state_dict['model_state_dict']
        
    soup_state_dict = OrderedDict()
    
    for key, tensor in base_state_dict.items():
        soup_state_dict[key] = torch.zeros_like(tensor, dtype=torch.float32)

    for ckpt in checkpoints:
        state_dict = torch.load(ckpt, map_location='cpu', weights_only=True)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        for key in soup_state_dict.keys():
            soup_state_dict[key] += state_dict[key].to(torch.float32)

    num_models = len(checkpoints)
    for key in soup_state_dict.keys():
        soup_state_dict[key] = (soup_state_dict[key] / num_models).to(base_state_dict[key].dtype)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(soup_state_dict, output_path)
    print(f"\n[+] ¡Model soup creado con éxito! Guardado en: {output_path}")

if __name__ == "__main__":
    MODEL_PREFIX = cfg.model_cfg.model_name
    OUTPUT_PATH = os.path.join(cfg.output_dir, f"{MODEL_PREFIX}_soup_5_epochs.pth")

    make_model_soup(cfg.output_dir, MODEL_PREFIX, OUTPUT_PATH)