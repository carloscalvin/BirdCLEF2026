import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.configs.config import cfg

def analyze_pseudo(thresh: float | None = None, plot: bool = True):
    if thresh is None:
        thresh = cfg.pseudo_threshold

    print(f"Loading pseudo-labels from {cfg.pseudo_soundscape_labels_path}")
    df = pd.read_csv(cfg.pseudo_soundscape_labels_path)

    non_class_cols = ['row_id']
    class_cols = [c for c in df.columns if c not in non_class_cols]
    class_cols = sorted(class_cols)
    print(f"Found {len(class_cols)} classes and {len(df)} rows")

    probs = df[class_cols]
    max_probs = probs.max(axis=1)
    print("\nOverall prediction statistics:")
    print(max_probs.describe())

    mask = (probs >= thresh).any(axis=1)
    kept = df[mask].copy()
    print(f"\nThreshold {thresh}: kept {len(kept)} rows ({len(df)-len(kept)} filtered)")

    class_counts = (kept[class_cols] >= thresh).sum().sort_values(ascending=False)
    
    positive_classes = class_counts[class_counts > 0]
    num_positive = len(positive_classes)

    print(f"\n=========================================")
    print(f"Total de clases positivas (>= threshold): {num_positive} de {len(class_cols)}")
    print(f"=========================================")
    print("\nCantidad de pseudo-etiquetas por CADA clase positiva:")

    with pd.option_context('display.max_rows', None):
        print(positive_classes)
    print("=========================================\n")

    if plot:
        plt.figure()
        max_probs.hist(bins=50)
        plt.title('Distribution of max probability per row')
        plt.xlabel('max probability')
        plt.ylabel('count')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        positive_classes.plot.bar()
        plt.title('Number of positive pseudo-labels per class')
        plt.ylabel('count')
        plt.tight_layout()
        plt.show()

    return class_counts, max_probs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze pseudo-label CSV distribution"
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=None,
        help='override the configured pseudo threshold',
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='do not display plots',
    )
    args = parser.parse_args()

    analyze_pseudo(thresh=args.threshold, plot=not args.no_plot)
