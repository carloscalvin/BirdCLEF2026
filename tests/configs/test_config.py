from src.configs.config import cfg
import torch
from types import SimpleNamespace

def test_config_structure():
    assert hasattr(cfg, "project_name")
    assert hasattr(cfg, "data_dir")
    assert isinstance(cfg.seed, int)

def test_device_config():
    assert isinstance(cfg.device, torch.device)

def test_model_config_exists():
    assert hasattr(cfg, "model_cfg")
    assert isinstance(cfg.model_cfg, SimpleNamespace)
    assert cfg.model_cfg.model_name == "tf_efficientnet_b0_ns"

def test_mixup_hyperparams_present():
    # Garantiza que el experimento de soft-mixup + cutmix temporal queda trazado.
    assert hasattr(cfg, "mixup_prob")
    assert hasattr(cfg, "mixup_alpha")
    assert hasattr(cfg, "mixup_warmup_epochs")
    assert hasattr(cfg, "mixup_cutmix_prob")
    assert hasattr(cfg, "mixup_force_dominant")
    assert 0.0 <= cfg.mixup_prob <= 1.0
    assert cfg.mixup_alpha > 0.0
    assert cfg.mixup_warmup_epochs >= 0
    assert 0.0 <= cfg.mixup_cutmix_prob <= 1.0
    assert isinstance(cfg.mixup_force_dominant, bool)

def test_pseudo_soft_labels_flag_present():
    assert hasattr(cfg, "pseudo_use_soft_labels")
    assert isinstance(cfg.pseudo_use_soft_labels, bool)