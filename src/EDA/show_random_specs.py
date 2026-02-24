import os
import sys
import random

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.configs.config import cfg

def show_random_specs(mode: str, count: int = 10) -> None:
    directory_map = {
        'train': cfg.preprocess_train_dir,
        'val': cfg.preprocess_val_dir,
        'pseudo': cfg.preprocess_pseudo_dir,
    }

    if mode not in directory_map:
        raise ValueError(f"mode must be one of {list(directory_map.keys())}")

    spec_dir = directory_map[mode]
    if not os.path.exists(spec_dir):
        raise FileNotFoundError(f"directory does not exist: {spec_dir}")

    all_files = [os.path.join(spec_dir, f) for f in os.listdir(spec_dir) if f.lower().endswith('.npy')]
    if not all_files:
        raise FileNotFoundError(f"no numpy specs found in {spec_dir}")

    selected = random.sample(all_files, min(count, len(all_files)))

    for path in selected:
        spec = np.load(path)
        plt.figure(figsize=(6, 4))
        plt.imshow(spec, aspect='auto', origin='lower', interpolation='nearest')
        plt.title(os.path.basename(path))
        plt.colorbar(label='amplitude')
        plt.xlabel('time frames')
        plt.ylabel('mel bins')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Show random mel-specs from preprocessing directories"
    )
    parser.add_argument(
        'mode',
        choices=['train', 'val', 'pseudo'],
        help='which set to sample from',
    )
    parser.add_argument(
        '--count', '-n',
        type=int,
        default=10,
        help='number of random specs to display',
    )
    args = parser.parse_args()

    show_random_specs(args.mode, args.count)
