import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import wandb
from tqdm import tqdm
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.configs.config import cfg
from src.data.dataset import BirdDataset
from src.data.transforms import get_transforms
from src.models.model import BirdModel, BirdSEDModel, ModelEMA
from src.modules.metrics import macro_auc
from src.modules.losses import BCEFocalLoss
from src.data.augs import Mixup

def build_model(num_classes: int):
    mcfg = cfg.model_cfg
    if getattr(mcfg, 'use_sed', False):
        model = BirdSEDModel(
            model_name=mcfg.model_name,
            num_classes=num_classes,
            pretrained=mcfg.pretrained,
            use_gru=getattr(mcfg, 'sed_use_gru', True),
            gru_hidden=getattr(mcfg, 'sed_gru_hidden', 256),
            gru_layers=getattr(mcfg, 'sed_gru_layers', 2),
            gru_dropout=getattr(mcfg, 'sed_gru_dropout', 0.2),
        )
        arch = f"BirdSEDModel (GRU={'on' if getattr(mcfg, 'sed_use_gru', True) else 'off'})"
    else:
        model = BirdModel(
            model_name=mcfg.model_name,
            num_classes=num_classes,
            pretrained=mcfg.pretrained,
        )
        arch = "BirdModel (GeM baseline)"
    print(f"Arquitectura: {arch}")
    return model

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, ema_model, loader, pseudo_loader, pseudo_iter, optimizer, scheduler, criterion, device, scaler, mixup_fn, epoch=0):
    model.train()
    running_loss = 0.0
    n_mixed = 0
    n_pseudo_mixed = 0
    n_total = 0

    if mixup_fn is not None and hasattr(mixup_fn, "set_epoch"):
        mixup_fn.set_epoch(epoch)

    pbar = tqdm(loader, desc=f"Train (ep {epoch+1})", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        x_pseudo, y_pseudo = None, None
        is_pseudo_mix = False

        if pseudo_loader is not None and np.random.rand() < cfg.pseudo_mixup_ratio:
            is_pseudo_mix = True
            try:
                x_pseudo, y_pseudo = next(pseudo_iter)
            except StopIteration:
                pseudo_iter = iter(pseudo_loader)
                x_pseudo, y_pseudo = next(pseudo_iter)

            x_pseudo = x_pseudo.to(device)
            y_pseudo = y_pseudo.to(device)

        n_total += images.size(0)
        original_targets = targets
        images, targets = mixup_fn(images, targets, x_pseudo, y_pseudo, is_pseudo_mix)
        if targets is not original_targets:
            n_mixed += images.size(0)
            if is_pseudo_mix:
                n_pseudo_mixed += images.size(0)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        
        if ema_model:
            ema_model.update(model)
   
        running_loss += loss.item() * images.size(0)

        if scheduler is not None:
            scheduler.step()
            
        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(loader.dataset)
    mix_frac = (n_mixed / n_total) if n_total > 0 else 0.0
    pseudo_mix_frac = (n_pseudo_mixed / n_total) if n_total > 0 else 0.0
    return epoch_loss, pseudo_iter, mix_frac, pseudo_mix_frac

@torch.no_grad()
def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="Valid (Teacher)", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(outputs)
        
        all_preds.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds)

    all_targets = np.concatenate(all_targets)

    epoch_auc = macro_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc

def run_training():
    print(f"\n--- Iniciando Entrenamiento BirdCLEF Teacher Student ---")

    class_names = pd.read_csv(cfg.classes_order_path, header=None)[0].tolist()
    num_classes = len(class_names)
    print(f"Clases cargadas: {num_classes}")

    df_train = pd.read_csv(os.path.join(cfg.data_dir, "train_processed.csv"))
    df_val = pd.read_pickle(cfg.val_processed_path)

    train_ds = BirdDataset(
        df_train, 
        cfg.preprocess_train_dir, 
        transform=get_transforms('train'), 
        mode='train',
        class_names=class_names
    )
    
    val_ds = BirdDataset(
        df_val, 
        cfg.preprocess_val_dir, 
        transform=get_transforms('valid'), 
        mode='valid',
        class_names=class_names
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        num_workers=cfg.num_workers, 
        pin_memory=True, 
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers, 
        pin_memory=True
    )

    pseudo_loader = None
    pseudo_iter = None
    if cfg.use_pseudo_labels:
        pseudo_df = pd.read_pickle(cfg.pseudo_processed_path)
        pseudo_ds = BirdDataset(
            pseudo_df,
            cfg.preprocess_pseudo_dir,
            transform=get_transforms('train'),
            mode='pseudo',
            class_names=class_names,
            pseudo_threshold=cfg.pseudo_threshold,
            pseudo_use_soft_labels=getattr(cfg, 'pseudo_use_soft_labels', False),
        )
        pseudo_loader = DataLoader(
            pseudo_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True
        )
        pseudo_iter = iter(pseudo_loader)
        print(f"Inyector pseudo activado con {len(pseudo_ds)} muestras.")

    print(f"Creando modelo {cfg.model_cfg.model_name} para {num_classes} clases...")
    model = build_model(num_classes)
    model.to(cfg.device)

    ema_model = ModelEMA(model, decay=cfg.model_cfg.ema_decay, device=cfg.device)
    ema_model.set(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(f"Usando Focal Loss (Gamma={cfg.loss_gamma:.4f}, Alpha={cfg.loss_alpha:.4f})")
    criterion = BCEFocalLoss(
        alpha=cfg.loss_alpha, 
        gamma=cfg.loss_gamma,
        bce_weight=cfg.loss_bce_weight,
        focal_weight=cfg.loss_focal_weight
    )
    scaler = GradScaler(enabled=cfg.use_amp)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.min_lr)

    mixup_warmup = getattr(cfg, 'mixup_warmup_epochs', 0)
    mixup_cutmix = getattr(cfg, 'mixup_cutmix_prob', 0.0)
    mixup_force_dom = getattr(cfg, 'mixup_force_dominant', False)
    print(
        f"Configurando Mixup: Prob={cfg.mixup_prob}, Alpha={cfg.mixup_alpha}, "
        f"CutMix(t)={mixup_cutmix}, Warmup={mixup_warmup}, ForceDominant={mixup_force_dom}, "
        f"PseudoSoft={getattr(cfg, 'pseudo_use_soft_labels', False)}"
    )
    mixup_fn = Mixup(
        mixup_prob=cfg.mixup_prob,
        alpha=cfg.mixup_alpha,
        cutmix_prob=mixup_cutmix,
        warmup_epochs=mixup_warmup,
        force_dominant=mixup_force_dom,
    )

    wandb.init(
        project=cfg.project_name,
        name=cfg.exp_name,
        config=cfg.__dict__
    )
    
    for epoch in range(cfg.epochs):
        train_loss, pseudo_iter, mix_frac, pseudo_mix_frac = train_one_epoch(
            model, ema_model, train_loader, pseudo_loader, pseudo_iter,
            optimizer, scheduler, criterion, cfg.device, scaler, mixup_fn,
            epoch=epoch,
        )

        val_loss, val_auc = valid_one_epoch(ema_model.module, val_loader, criterion, cfg.device)

        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
            f"AUC: {val_auc:.4f} | mix={mix_frac:.2f} (pseudo={pseudo_mix_frac:.2f})"
        )

        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "lr": scheduler.get_last_lr()[0],
            "mixup_active_frac": mix_frac,
            "mixup_pseudo_frac": pseudo_mix_frac,
        })

        if epoch >= (cfg.epochs - cfg.epochs_to_save):
                    save_path = os.path.join(cfg.output_dir, f"{cfg.model_cfg.model_name}_epoch_{epoch+1}.pth")
                    os.makedirs(cfg.output_dir, exist_ok=True)
                    torch.save(ema_model.module.state_dict(), save_path)
                    print(f" [S] Modelo del epoch {epoch+1}, AUC: {val_auc:.4f} guardado para el ensamble!")
            
    wandb.finish()
    print("Entrenamiento finalizado.")

if __name__ == "__main__":
    wandb.login()
    seed_everything(cfg.seed)
    run_training()