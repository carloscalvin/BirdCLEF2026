from types import SimpleNamespace
import torch
import os

cfg = SimpleNamespace(**{})

cfg.project_name = "BirdCLEF2026"
cfg.num_workers = 0
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.seed = 42
cfg.fast_dev_run = False

cfg.target_year = "2025" 
cfg.dataset_slug = f"birdclef-{cfg.target_year}"
cfg.data_dir = "dataset/"
cfg.dataset_root = os.path.join(cfg.data_dir, cfg.dataset_slug)

cfg.train_audio_dir = os.path.join(cfg.dataset_root, "train_audio")
cfg.train_soundscapes_dir = os.path.join(cfg.dataset_root, "train_soundscapes")
cfg.test_soundscapes_dir = os.path.join(cfg.dataset_root, "test_soundscapes")

cfg.train_csv_path = os.path.join(cfg.dataset_root, "train.csv")
cfg.taxonomy_csv_path = os.path.join(cfg.dataset_root, "taxonomy.csv")
cfg.sample_submission_path = os.path.join(cfg.dataset_root, "sample_submission.csv")

cfg.train_preprocessed_dir = f"{cfg.data_dir}Train/preprocessed/"
cfg.output_dir = "outputs/"

cfg.fold = 0
cfg.n_folds = 5
cfg.batch_size = 64
cfg.epochs = 50
cfg.lr = 1e-3

model_cfg = SimpleNamespace(**{})
model_cfg.model_name = "BirdClefModel"
cfg.model_cfg = model_cfg