import os
import sys
import librosa
import contextlib
import numpy as np
import math
import random
import soundfile as sf
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Generator, Union

@contextlib.contextmanager
def suppress_c_stderr() -> Generator[None, None, None]:
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, stderr_fd)
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)

def list_audio_files(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    audio_files = []
    for root, _, files in os.walk(directory):
        for fn in files:
            if extensions is None or os.path.splitext(fn)[1].lower() in extensions:
                audio_files.append(os.path.join(root, fn))
    return audio_files

def get_duration(path: str) -> float:
    with suppress_c_stderr():
        y, sr = librosa.load(path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
    return duration

def get_sample_rate(path: str) -> int:
    with suppress_c_stderr():
        _, sr = librosa.load(path, sr=None)
    return sr

def resample(input_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    with suppress_c_stderr():
        y, sr_orig = librosa.load(input_path, sr=None)
        y_res = librosa.resample(y, orig_sr=sr_orig, target_sr=target_sr)
    return y_res, sr_orig

def get_audio(path: str, sr: int, is_mono: bool = True) -> Tuple[np.ndarray, int]:
    with suppress_c_stderr():
        y, sr_loaded = librosa.load(path, sr=sr, mono=is_mono)
    return y, sr_loaded

def save_audio(out_path: str, y: np.ndarray, sr: int) -> None:
    sf.write(out_path, y, sr)

def resample_and_save(input_path: str, target_sr: int, output_dir: str) -> str:
    y_res, sr_orig = resample(input_path, target_sr)
    fn = os.path.basename(input_path)
    out_path = os.path.join(output_dir, f"resampled_{target_sr}_{fn}")
    save_audio(out_path, y_res, target_sr)
    print(f"{input_path} ({sr_orig} Hz) → {out_path} ({target_sr} Hz)")
    return out_path

def pick_random_file(directory: str, extensions: Optional[List[str]] = None) -> str:
    files = list_audio_files(directory, extensions=extensions)
    if not files:
        raise FileNotFoundError(f"No audio files found in {directory!r}")
    return random.choice(files)

def calculate_rms(y: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(y))) + 1e-8

def mix_multiple_audios(
    bird_chunks: List[Optional[np.ndarray]], 
    noise_chunk: Optional[np.ndarray], 
    min_snr_db: float, 
    max_snr_db: float, 
    duration: int = 5, 
    sr: int = 32_000
) -> Optional[np.ndarray]:
    if noise_chunk is None:
        return None
    if not bird_chunks:
        return noise_chunk
    samples_per_chunk = duration*sr
    noise_chunk = noise_chunk[:samples_per_chunk]
    if len(noise_chunk) < samples_per_chunk:
        noise_chunk = np.pad(noise_chunk, (0, samples_per_chunk - len(noise_chunk)), mode='constant')
    rms_noise = calculate_rms(noise_chunk)
    mixed_signal = noise_chunk.copy().astype(np.float32)
    for y_bird in bird_chunks:
        if y_bird is None: continue
        y_bird = y_bird[:samples_per_chunk]
        if len(y_bird) < samples_per_chunk:
            y_bird = np.pad(y_bird, (0, samples_per_chunk - len(y_bird)), mode='constant')
        rms_bird = calculate_rms(y_bird)
        snr_db = random.uniform(min_snr_db, max_snr_db)
        snr_linear = 10**(snr_db / 10.0)
        target_rms_bird = rms_noise * math.sqrt(snr_linear)
        amp_factor = target_rms_bird / rms_bird
        mixed_signal += (y_bird * amp_factor)
    max_amp = np.max(np.abs(mixed_signal))
    if max_amp > 1.0:
        mixed_signal /= max_amp
    elif max_amp == 0:
        pass
    return mixed_signal.astype(np.float32)

def plot_mel_spectrogram(
    y: np.ndarray, 
    sr: int, 
    title: Optional[str] = None, 
    n_mels: int = 128, 
    fmax: Optional[int] = None
) -> None:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        fmax=fmax
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(8, 4))
    plt.imshow(S_db, aspect='auto', origin='lower')
    plt.title(title or "Mel spectrogram")
    plt.xlabel("Time frames")
    plt.ylabel("Mel bands")
    plt.tight_layout()
    plt.show()

def compare_two_mel_spectrograms(
    y1: np.ndarray, 
    y2: np.ndarray, 
    sr: int, 
    titles: Tuple[str, str] = ('Signal 1', 'Signal 2'), 
    n_mels: int = 128, 
    fmax: Optional[int] = None
) -> None:
    S1 = librosa.feature.melspectrogram(y=y1, sr=sr, n_mels=n_mels, fmax=fmax)
    S2 = librosa.feature.melspectrogram(y=y2, sr=sr, n_mels=n_mels, fmax=fmax)
    S1_db = librosa.power_to_db(S1, ref=np.max)
    S2_db = librosa.power_to_db(S2, ref=np.max)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].imshow(S1_db, aspect='auto', origin='lower')
    axes[0].set_title(titles[0])
    axes[0].set_ylabel('Mel bands')

    axes[1].imshow(S2_db, aspect='auto', origin='lower')
    axes[1].set_title(titles[1])
    axes[1].set_ylabel('Mel bands')
    axes[1].set_xlabel('Time frames')

    plt.tight_layout()
    plt.show()

def mix_multiple_chunks(
    clean_dir: str, 
    noise_dir: str, 
    exts: Optional[List[str]], 
    min_snr_db: float = 0, 
    max_snr_db: float = 15, 
    sr: int = 32_000, 
    duration: int = 5, 
    bird_chunks_number: int = 5
) -> Optional[np.ndarray]:
    bird_chunks = []
    for _ in range(bird_chunks_number):
        bird_chunk_path = pick_random_file(clean_dir, extensions=exts)
        bird_chunk, _ = resample(bird_chunk_path, sr)
        bird_chunks.append(bird_chunk)
    noise_chunk_path = pick_random_file(noise_dir, extensions=exts)
    noise_chunk, _ = resample(noise_chunk_path, sr)

    mixed_signal = mix_multiple_audios(
        bird_chunks,
        noise_chunk,
        min_snr_db=min_snr_db,
        max_snr_db=max_snr_db,
        duration=duration,
        sr=sr
    )
    return mixed_signal