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