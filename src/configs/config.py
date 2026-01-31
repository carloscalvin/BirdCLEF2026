from types import SimpleNamespace
import torch

cfg = SimpleNamespace(**{})

cfg.project_name = "BirdCLEF2026"
cfg.num_workers = 0
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.seed = 42
cfg.fast_dev_run = False

cfg.data_dir = "dataset/"
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