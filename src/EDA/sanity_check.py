import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.configs.config import cfg
from src.preprocessing import utils

def run_sanity_check():
    audio_dir = os.path.join(cfg.data_dir, "birdclef-2025", "train_audio", "blbgra1") 
    file_path = utils.pick_random_file(audio_dir, extensions=['.ogg'])

    print(f"[*] File: {os.path.basename(file_path)}")

    original_duration = utils.get_duration(file_path)
    original_sr = utils.get_sample_rate(file_path)
    print(f"    Duration: {original_duration:.2f} secs")
    print(f"    Sample rate: {original_sr} Hz")

    y, _ = utils.get_audio(file_path, sr=None)
    print(f"    Min/Max: {y.min():.4f} / {y.max():.4f}")

    TARGET_SR = 32000
    print(f"\n Test sample {TARGET_SR} Hz...")
    y_res, sr_orig_check = utils.resample(file_path, target_sr=TARGET_SR)
    
    assert sr_orig_check == original_sr, "Error match"
    print(f"    New: {len(y_res)} samples")
    print(f"    Duration (len/sr): {len(y_res)/TARGET_SR:.2f} secs")


    print("\n[3] Test mix (bird + noise)...")
    noise = np.random.normal(0, 0.1, len(y_res)).astype(np.float32)

    y_mixed = utils.mix_multiple_audios(
        bird_chunks=[y_res],
        noise_chunk=noise,
        min_snr_db=5,
        max_snr_db=10,
        duration=int(original_duration),
        sr=TARGET_SR
    )

    if y_mixed is not None:
        print(f"    Mix: {y_mixed.min():.4f} / {y_mixed.max():.4f}")
    else:
        print("    [!] Mix error.")

    utils.compare_two_mel_spectrograms(
        y1=y_res,
        y2=y_mixed if y_mixed is not None else y_res,
        sr=TARGET_SR,
        titles=(
            f'Source ({TARGET_SR}Hz)', 
            'Mix'
        ),
        n_mels=128,
        fmax=16000
    )

    print("\n--- SANITY CHECK DONE ---")

if __name__ == "__main__":
    run_sanity_check()