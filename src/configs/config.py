from types import SimpleNamespace
import torch
import os

cfg = SimpleNamespace(**{})

cfg.project_name = "BirdCLEF2026_Pantanal"
cfg.exp_name = "tf_efficientnet_b0_ns_baseline_run2"
cfg.num_workers = 0
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.seed = 42
cfg.fast_dev_run = False

cfg.target_year = "2026"
cfg.dataset_slug = f"birdclef-{cfg.target_year}"
cfg.data_dir = "dataset/"
cfg.dataset_root = os.path.join(cfg.data_dir, cfg.dataset_slug)

cfg.train_audio_dir = os.path.join(cfg.dataset_root, "train_audio")
cfg.train_soundscapes_dir = os.path.join(cfg.dataset_root, "train_soundscapes")
cfg.test_soundscapes_dir = os.path.join(cfg.dataset_root, "test_soundscapes")

cfg.train_csv_path = os.path.join(cfg.dataset_root, "train.csv")
cfg.taxonomy_csv_path = os.path.join(cfg.dataset_root, "taxonomy.csv")
cfg.sample_submission_path = os.path.join(cfg.dataset_root, "sample_submission.csv")
cfg.teacher_preds_path = os.path.join(cfg.dataset_root, "val_soundscape.csv")
cfg.pseudo_soundscape_labels_path = os.path.join(cfg.dataset_root, "pseudo_soundscape.csv")
cfg.classes_order_path = os.path.join(cfg.dataset_root, "classes_order.csv")
cfg.val_processed_path = os.path.join(cfg.dataset_root, "val_processed.pkl")
cfg.pseudo_processed_path = os.path.join(cfg.dataset_root, "pseudo_processed.pkl")

cfg.output_dir = "outputs/"
cfg.sr = 32000
cfg.duration = 5
cfg.step = 1
cfg.n_mels = 224
cfg.fmin = 0
cfg.fmax = 16000
cfg.n_fft = 4096
cfg.hop_length = 512

cfg.preprocess_train_dir = os.path.join(cfg.data_dir, "train_specs")
os.makedirs(cfg.preprocess_train_dir, exist_ok=True)
cfg.preprocess_val_dir = os.path.join(cfg.data_dir, "val_soundscape_specs")
os.makedirs(cfg.preprocess_val_dir, exist_ok=True)
cfg.preprocess_pseudo_dir = os.path.join(cfg.data_dir, "pseudo_soundscape_specs")
os.makedirs(cfg.preprocess_pseudo_dir, exist_ok=True)

cfg.batch_size = 64
cfg.epochs = 50
cfg.n_folds = 5
cfg.lr = 1e-3
cfg.min_lr = 1e-6
cfg.weight_decay = 1e-4
cfg.max_grad_norm = 1.0
cfg.use_amp = True
cfg.loss_alpha=0.25
cfg.loss_gamma=2.0
cfg.loss_bce_weight = 0.6
cfg.loss_focal_weight = 1.4

cfg.mixup_prob = 1
cfg.mixup_alpha = 1.0

cfg.spec_aug_time_mask = 0
cfg.spec_aug_freq_mask = 0
cfg.spec_aug_prob = 0.0

cfg.gaussian_noise_prob = 0.2
cfg.gaussian_noise_limit = (0.5, 2.0)

model_cfg = SimpleNamespace(**{})
model_cfg.model_name = "tf_efficientnet_b0_ns"
model_cfg.pretrained = True
model_cfg.num_classes = 0
model_cfg.ema_decay = 0.999
cfg.model_cfg = model_cfg

cfg.apply_postprocess = False
cfg.post_top_k = 30
cfg.post_exponent = 2.0

cfg.use_sliding_window_infer = False
cfg.infer_overlaps = 2

cfg.apply_smoothing = True
cfg.smoothing_weights = (0.15, 0.70, 0.15)

cfg.reduce_noise = False
cfg.reduce_noise_prop_decrease = 0.8
cfg.reduce_noise_stationary = True

cfg.train_enrichment_threshold = 0.65

cfg.use_pseudo_labels = False
cfg.pseudo_threshold = 0.0
cfg.pseudo_mixup_ratio = 0.5