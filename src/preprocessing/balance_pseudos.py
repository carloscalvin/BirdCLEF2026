import argparse
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.configs.config import cfg


@dataclass(frozen=True)
class BalanceConfig:
    threshold: float
    val_fraction: float
    seed: int
    target_per_species: Optional[int]
    target_quantile: float
    min_per_species: int


def _get_class_columns(df: pd.DataFrame) -> Sequence[str]:
    non_class_cols = ["row_id"]
    class_cols = [c for c in df.columns if c not in non_class_cols]
    class_cols = sorted(class_cols)
    if not class_cols:
        raise ValueError("No class columns found in pseudo CSV.")
    return class_cols


def _assign_species_top_positive(
    probs: np.ndarray, class_cols: Sequence[str], threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each row, pick the top-1 class among those with prob >= threshold.
    Returns:
      species_idx: (N,) index into class_cols
      species_prob: (N,) corresponding max prob (>= threshold for kept rows)
    """
    pos = probs >= threshold
    species_choice_scores = np.where(pos, probs, -1.0)
    species_idx = species_choice_scores.argmax(axis=1)
    species_prob = species_choice_scores.max(axis=1)
    # species_prob < 0 means the row had no class >= threshold
    return species_idx, species_prob


def balance_pseudo_csv(
    input_csv: str,
    output_train_csv: str,
    output_val_csv: str,
    *,
    class_cols: Optional[Sequence[str]] = None,
    balance_cfg: Optional[BalanceConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a new pseudo-label dataset that is balanced by "species top-1",
    where each pseudo row is assigned to a single species: the most confident class
    among those above `threshold`.

    Then performs an 80/20 split stratified by that assigned species.
    """
    if balance_cfg is None:
        balance_cfg = BalanceConfig(
            threshold=cfg.pseudo_threshold,
            val_fraction=0.2,
            seed=cfg.seed,
            target_per_species=None,
            target_quantile=0.2,
            min_per_species=1,
        )

    df = pd.read_csv(input_csv)
    if class_cols is None:
        class_cols = _get_class_columns(df)

    probs = df[list(class_cols)].to_numpy(dtype=np.float32)

    rng = np.random.default_rng(balance_cfg.seed)

    species_idx, species_prob = _assign_species_top_positive(probs, class_cols, balance_cfg.threshold)
    mask_kept = species_prob >= 0
    df_kept = df.loc[mask_kept].copy().reset_index(drop=True)
    species_idx_kept = species_idx[mask_kept]

    if len(df_kept) == 0:
        raise ValueError(
            f"After filtering with threshold={balance_cfg.threshold}, no rows remain."
        )

    species_names_kept = np.array(class_cols, dtype=object)[species_idx_kept]
    unique_species, counts = np.unique(species_names_kept, return_counts=True)
    counts_dict = dict(zip(unique_species.tolist(), counts.tolist()))

    if balance_cfg.target_per_species is None:
        target = int(np.quantile(counts[counts > 0], balance_cfg.target_quantile))
    else:
        target = int(balance_cfg.target_per_species)

    target = max(balance_cfg.min_per_species, target)
    print(
        f"[balance_pseudos] input rows={len(df)} kept rows={len(df_kept)} "
        f"species={len(unique_species)} target_per_species={target}"
    )

    # Sample per species (top-1 assignment -> each row belongs to exactly one bucket).
    selected_row_indices: list[int] = []
    for sp in unique_species.tolist():
        idxs = np.where(species_names_kept == sp)[0]
        n = len(idxs)
        if n == 0:
            continue
        if n >= target:
            chosen = rng.choice(idxs, size=target, replace=False)
        else:
            chosen = rng.choice(idxs, size=target, replace=True)
        selected_row_indices.extend(chosen.tolist())

    balanced_df = df_kept.loc[selected_row_indices].copy().reset_index(drop=True)
    balanced_species = species_names_kept[np.array(selected_row_indices, dtype=int)]
    balanced_df["_species_top"] = balanced_species

    # Stratified 80/20 split by _species_top
    train_indices: list[int] = []
    val_indices: list[int] = []

    for sp in unique_species.tolist():
        idxs = np.where(balanced_species == sp)[0]
        if len(idxs) == 0:
            continue
        rng.shuffle(idxs)
        n_val = int(round(len(idxs) * balance_cfg.val_fraction))
        if len(idxs) >= 2:
            n_val = min(max(1, n_val), len(idxs) - 1)
        else:
            n_val = 0

        val_idxs = idxs[:n_val]
        train_idxs = idxs[n_val:]
        val_indices.extend(val_idxs.tolist())
        train_indices.extend(train_idxs.tolist())

    train_df = balanced_df.loc[train_indices].drop(columns=["_species_top"]).reset_index(drop=True)
    val_df = balanced_df.loc[val_indices].drop(columns=["_species_top"]).reset_index(drop=True)

    # Keep identical column order as input
    train_df = train_df[df.columns]
    val_df = val_df[df.columns]

    os.makedirs(os.path.dirname(output_train_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_val_csv), exist_ok=True)
    train_df.to_csv(output_train_csv, index=False)
    val_df.to_csv(output_val_csv, index=False)

    print(
        f"[balance_pseudos] wrote train rows={len(train_df)} "
        f"val rows={len(val_df)} to CSV"
    )
    return train_df, val_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balance raw pseudo-label CSV and create an 80/20 stratified split."
    )
    parser.add_argument("--input-csv", default=cfg.pseudo_soundscape_labels_path_raw)
    parser.add_argument("--output-train-csv", default=cfg.pseudo_soundscape_labels_path)
    parser.add_argument("--output-val-csv", default=cfg.teacher_preds_path)
    parser.add_argument("--threshold", type=float, default=cfg.pseudo_threshold)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--target-per-species", type=int, default=None)
    parser.add_argument("--target-quantile", type=float, default=0.2)
    parser.add_argument("--min-per-species", type=int, default=1)

    args = parser.parse_args()

    balance_cfg = BalanceConfig(
        threshold=args.threshold,
        val_fraction=args.val_fraction,
        seed=args.seed,
        target_per_species=args.target_per_species,
        target_quantile=args.target_quantile,
        min_per_species=args.min_per_species,
    )

    balance_pseudo_csv(
        args.input_csv,
        args.output_train_csv,
        args.output_val_csv,
        balance_cfg=balance_cfg,
    )


if __name__ == "__main__":
    main()
