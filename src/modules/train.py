import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
import wandb
from tqdm import tqdm
import torch.nn.functional as F 
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.configs.config import cfg
from src.data.dataset import BirdDataset
from src.data.transforms import get_transforms
from src.data.augs import MixupCutmix
from src.models.model import BirdModel, ModelEMA

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, ema_model, loader, optimizer, scheduler, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    augmenter = MixupCutmix(
        mixup_prob=cfg.mixup_prob, 
        cutmix_prob=cfg.cutmix_prob, 
        alpha=cfg.mixup_alpha
    )

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        images, target_a, target_b, lam, type = augmenter(images, labels)
        
        with autocast(enabled=cfg.use_amp):
            outputs = model(images)
            
            if type != 'none':
                loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
            else:
                loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    
        if ema_model:
            ema_model.update(model)
   
        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        if type != 'none':
            correct += (lam * predicted.eq(target_a).float() + (1 - lam) * predicted.eq(target_b).float()).sum().item()
        else:
            correct += predicted.eq(labels).sum().item()

        if scheduler is not None:
            scheduler.step()
            
        pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="Valid", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        probs = F.softmax(outputs, dim=1)
        all_preds.append(probs.cpu().numpy())
        all_targets.append(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    epoch_auc = roc_auc_score(all_targets, all_preds, multi_class='ovr', average='macro')

    return epoch_loss, epoch_auc

def run_fold(fold, df, train_files, val_files):
    print(f"\n--- Iniciando Fold: {fold+1}/{cfg.n_folds} ---")
    print(f"Train Files: {len(train_files)} | Val Files: {len(val_files)}")

    df_train = df[df['filename'].isin(train_files)].reset_index(drop=True)
    df_val = df[df['filename'].isin(val_files)].reset_index(drop=True)

    train_ds = BirdDataset(
        df_train, 
        cfg.preprocess_train_dir, 
        transform=get_transforms('train'),
        mode='train'
    )

    val_ds = BirdDataset(
        df_val, 
        cfg.preprocess_train_dir, 
        transform=get_transforms('valid'),
        mode='valid'
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

    model = BirdModel(cfg.model_cfg.model_name, cfg.model_cfg.num_classes, pretrained=cfg.model_cfg.pretrained)
    model.to(cfg.device)

    ema_model = ModelEMA(model, decay=cfg.model_cfg.ema_decay, device=cfg.device)
    ema_model.set(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=cfg.use_amp)

    total_steps = len(train_loader) * cfg.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps, 
        eta_min=cfg.min_lr
    )

    run_name = f"{cfg.model_cfg.model_name}_fold{fold+1}"
    wandb.init(
        project=cfg.project_name, 
        name=run_name, 
        group=cfg.exp_name,
        config=cfg.__dict__,
        reinit=True
    )
    
    best_auc = 0.0

    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(
            model, ema_model, train_loader, optimizer, scheduler, criterion, cfg.device, scaler
        )

        val_loss, val_auc = valid_one_epoch(ema_model.module, val_loader, criterion, cfg.device)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} AUC: {val_auc:.4f}")

        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "lr": scheduler.get_last_lr()[0]
        })

        if val_auc > best_auc:
            best_auc = val_auc
            save_name = f"{cfg.model_cfg.model_name}_fold{fold+1}_best.pth"
            os.makedirs(cfg.output_dir, exist_ok=True)
            save_path = os.path.join(cfg.output_dir, save_name)
            torch.save(ema_model.module.state_dict(), save_path)
            print(f" [S] Mejor modelo guardado (AUC: {best_auc:.4f})")
            
    wandb.finish()

    del model, ema_model, optimizer, scheduler, scaler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_auc

if __name__ == "__main__":
    wandb.login()
    seed_everything(cfg.seed)

    processed_csv_path = os.path.join(cfg.data_dir, "train_processed.csv")
    if not os.path.exists(processed_csv_path):
        print("ERROR: No se encuentra train_processed.csv. Ejecuta preprocess.py primero.")
        exit()

    print(f"Leyendo CSV Procesado: {processed_csv_path}")
    df = pd.read_csv(processed_csv_path)

    num_classes = df['primary_label'].nunique()
    cfg.model_cfg.num_classes = num_classes
    print(f"Detectadas {num_classes} clases.")

    unique_files_df = df.drop_duplicates('filename')[['filename', 'primary_label']].reset_index(drop=True)
    
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(unique_files_df, unique_files_df['primary_label'])):
        train_files = unique_files_df.loc[train_idx, 'filename'].values
        val_files = unique_files_df.loc[val_idx, 'filename'].values
        
        acc = run_fold(fold, df, train_files, val_files)
        fold_scores.append(acc)

    print("\n" + "="*40)
    print(f"CV RESULTS")
    print(f"Scores: {fold_scores}")
    print(f"Average Accuracy: {np.mean(fold_scores):.4f}")
    print("="*40)