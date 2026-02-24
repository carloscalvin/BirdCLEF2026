import os
import pandas as pd
import numpy as np
from src.preprocessing.preprocess_soundscapes import preprocess_soundscapes
from src.configs.config import cfg


def _make_dummy_preds(csv_path):
    df = pd.DataFrame({
        "row_id": ["file_0_5", "file_1_5", "file_2_5"],
        "classA": [0.1, 0.6, 0.4],
        "classB": [0.2, 0.3, 0.5],
    })
    df.to_csv(csv_path, index=False)


def test_pseudo_threshold_filtering(tmp_path, monkeypatch):
    csv_path = tmp_path / "preds.csv"
    _make_dummy_preds(csv_path)

    preprocess_dir = tmp_path / "specs"
    processed_path = tmp_path / "processed.pkl"

    monkeypatch.setattr(os.path, "exists", lambda x: True)

    monkeypatch.setattr(
        "src.preprocessing.preprocess_soundscapes.librosa.load",
        lambda *args, **kwargs: (np.zeros(cfg.sr * cfg.duration), cfg.sr),
    )

    monkeypatch.setattr(
        "src.preprocessing.preprocess_soundscapes.compute_melspec",
        lambda y, sr: np.zeros((cfg.n_mels, 10), dtype=np.float32),
    )

    cfg.pseudo_threshold = 0.5

    preprocess_soundscapes(str(csv_path), str(preprocess_dir), str(processed_path), pseudo=True)

    df_out = pd.read_pickle(str(processed_path))
    assert len(df_out) == 2
    assert set(df_out["row_id"]) == {"file_1_5", "file_2_5"}
