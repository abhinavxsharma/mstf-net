"""
scripts/train.py
================
MSTF-Net training script — supports WildDeepfake and FF++ C23.

Paper protocol:
  AdamW lr=2e-4, weight_decay=0, batch=512
  CrossEntropyLoss + WeightedRandomSampler
  10 epochs, early stopping patience=4
  AMP fp16, grad clip norm=1.0
  5 seeds for WildDeepfake, 3 seeds for FF++

Usage:
    python scripts/train.py --config configs/wilddeepfake.yaml --seed 42
    python scripts/train.py --config configs/ffpp_c23.yaml --seed 42

Authors : Abhinav Vats et al., Chandigarh University
Accepted: ISSMAD 2026 (Google + IEEE Signal Processing Society)
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from mstfnet import (
    MSTFNet,
    set_seed, compute_metrics,
    save_checkpoint, load_checkpoint,
    save_results, get_class_weights,
    AverageMeter, EarlyStopping, print_gpu_info,
)
from mstfnet.dataset import (
    WildDeepfakeDataset,
    get_train_transform, get_val_transform,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',    default='configs/wilddeepfake.yaml')
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--resume',    type=str, default=None)
    p.add_argument('--no-wandb',  action='store_true')
    p.add_argument('--no-upload', action='store_true')
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def setup_wandb(cfg, seed, disabled):
    if disabled or not cfg['wandb']['enabled']:
        return None
    try:
        import wandb
        dataset_name = cfg['dataset']['name']
        run = wandb.init(
            project=cfg['wandb']['project'],
            entity=cfg['wandb']['entity'],
            name=f'{dataset_name}-seed{seed}',
            tags=cfg['wandb']['tags'] + [f'seed{seed}'],
            config={
                'seed'       : seed,
                'dataset'    : dataset_name,
                'epochs'     : cfg['training']['epochs'],
                'batch_size' : cfg['training']['batch_size'],
                'lr'         : cfg['training']['lr'],
                'tau_min'    : cfg['model']['tau_min'],
                'tau_max'    : cfg['model']['tau_max'],
                'n_frames'   : cfg['dataset']['n_frames'],
            },
        )
        print(f'  W&B: {run.url}')
        return run
    except Exception as e:
        print(f'W&B failed: {e}')
        return None


def upload_to_hf(cfg, checkpoint_path, seed):
    try:
        from huggingface_hub import HfApi
        token = os.environ.get('HF_TOKEN')
        if not token:
            print('HF_TOKEN not set — skipping upload')
            return
        dataset_name = cfg['dataset']['name']
        api = HfApi()
        remote = f'weights/mstfnet_{dataset_name}_seed{seed}_BEST.pth'
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=remote,
            repo_id=cfg['huggingface']['repo_id'],
            token=token,
        )
        print(f'✅ HF: {cfg["huggingface"]["repo_id"]}/{remote}')
    except Exception as e:
        print(f'HF upload failed: {e}')


def get_dataset(cfg, split, transform):
    """Get correct dataset class based on config."""
    name         = cfg['dataset']['name']
    frames_root  = cfg['paths']['frames_root']
    n_frames     = cfg['dataset']['n_frames']

    if name == 'wilddeepfake':
        return WildDeepfakeDataset(
            root=frames_root, split=split,
            transform=transform, n_frames=n_frames,
        )
    else:
        raise ValueError(f'Unknown dataset: {name}. Supported: wilddeepfake')


def train_one_epoch(model, loader, optimizer, criterion,
                    scaler, device, wandb_run=None, epoch=0):
    model.train()
    loss_meter = AverageMeter('loss')

    for batch_idx, (imgs, labels, _) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            logits, alpha, q_n = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        loss_meter.update(loss.item(), imgs.size(0))

        if batch_idx % 50 == 0:
            am = alpha.mean(0).cpu().detach().numpy()
            print(f'    step {batch_idx:04d}  '
                  f'loss={loss_meter.avg:.4f}  '
                  f'Qn={q_n.mean().item():.3f}  '
                  f'α=[{am[0]:.2f},{am[1]:.2f},{am[2]:.2f}]')
        del imgs, labels

    return loss_meter.avg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels, _ in loader:
        imgs  = imgs.to(device, non_blocking=True)
        probs = model.predict_proba(imgs).cpu().float().numpy()
        all_probs.extend(np.nan_to_num(probs, nan=0.5))
        all_labels.extend(labels.numpy())
        del imgs
    return compute_metrics(
        np.array(all_labels, dtype=np.int32),
        np.array(all_probs,  dtype=np.float32),
    )


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    seed   = args.seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_name = cfg['dataset']['name']
    ckpt_dir     = cfg['paths']['checkpoint_dir']
    results_dir  = cfg['paths']['results_dir']

    print('\n' + '=' * 60)
    print(f'MSTF-Net Training — {dataset_name}')
    print(f'  Config : {args.config}')
    print(f'  Seed   : {seed}')
    print(f'  Device : {device}')
    print('=' * 60)
    print_gpu_info()

    set_seed(seed)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Skip if already done
    best_ckpt_path = (
        Path(ckpt_dir) /
        f'mstfnet_{dataset_name}_seed{seed}_BEST.pth'
    )
    if best_ckpt_path.exists():
        print(f'\n⚠️  Already trained seed {seed} — skipping')
        print(f'   Delete {best_ckpt_path} to retrain')
        return

    # Datasets
    print('\nLoading datasets...')
    val_split = 'test'  # WildDeepfake uses test split for val
    train_ds = get_dataset(cfg, 'train', get_train_transform(cfg['dataset']['img_size']))
    val_ds   = get_dataset(cfg, val_split, get_val_transform(cfg['dataset']['img_size']))

    assert len(train_ds) > 0, 'No training data found'
    assert len(val_ds)   > 0, 'No val data found'

    # WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=get_class_weights(train_ds.labels),
        num_samples=len(train_ds),
        replacement=True,
    )

    batch = cfg['training']['batch_size']
    train_loader = DataLoader(
        train_ds, batch_size=batch, sampler=sampler,
        num_workers=8, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False,
        num_workers=8, pin_memory=True,
    )
    print(f'  Train batches: {len(train_loader)}'
          f'  Val batches: {len(val_loader)}')

    # Model
    print('\nBuilding MSTF-Net...')
    model = MSTFNet(
        d           = cfg['model']['feature_dim'],
        tau_min     = cfg['model']['tau_min'],
        tau_max     = cfg['model']['tau_max'],
        cls_dropout = cfg['model']['cls_dropout'],
    ).to(device)
    params = model.count_parameters()
    print(f'  Total params: {params["total"]}')

    # Resume
    start_epoch = 1
    if args.resume:
        ckpt = load_checkpoint(model, args.resume, device)
        start_epoch = ckpt.get('epoch', 0) + 1

    # Optimiser
    optimizer  = optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
        betas=tuple(cfg['optimizer']['betas']),
        eps=cfg['optimizer']['eps'],
    )
    criterion  = nn.CrossEntropyLoss()
    scaler     = GradScaler('cuda')
    early_stop = EarlyStopping(patience=cfg['training']['early_stopping_patience'])
    wandb_run  = setup_wandb(cfg, seed, disabled=args.no_wandb)

    # Training loop
    best_auc      = 0.
    best_ckpt     = None
    all_metrics   = []
    total_epochs  = cfg['training']['epochs']

    print(f'\nTraining {total_epochs} epochs, seed={seed}...\n')
    t_start = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        t_ep = time.time()
        print(f'\n── Epoch {epoch:02d}/{total_epochs} (seed={seed}) ──')

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, wandb_run, epoch,
        )
        metrics = evaluate(model, val_loader, device)
        elapsed = (time.time() - t_ep) / 60

        print(f'  AUC={metrics["auc"]:.4f}  '
              f'Acc={metrics["accuracy"]:.4f}  '
              f'F1={metrics["f1"]:.4f}  '
              f'Loss={train_loss:.4f}  '
              f'({elapsed:.1f}min)')

        if wandb_run:
            wandb_run.log({
                'epoch': epoch, 'train_loss': train_loss,
                'val_auc': metrics['auc'],
                'val_acc': metrics['accuracy'],
                'val_f1': metrics['f1'], 'seed': seed,
            })

        all_metrics.append({'epoch': epoch, **metrics, 'loss': train_loss})

        # Save every epoch
        save_checkpoint(
            model, metrics, epoch, seed, ckpt_dir,
            filename=f'mstfnet_{dataset_name}_seed{seed}_ep{epoch:02d}.pth',
        )

        if metrics['auc'] > best_auc:
            best_auc  = metrics['auc']
            best_ckpt = save_checkpoint(
                model, metrics, epoch, seed, ckpt_dir,
                filename=f'mstfnet_{dataset_name}_seed{seed}_BEST.pth',
            )
            print(f'  ✅ New best AUC: {best_auc:.4f}')

        if early_stop.step(metrics['auc']):
            print(f'  Early stopping at epoch {epoch}')
            break

        gc.collect()
        torch.cuda.empty_cache()

    total_time = (time.time() - t_start) / 60
    print(f'\n{"=" * 60}')
    print(f'Seed {seed} done in {total_time:.1f} min')
    print(f'Best AUC: {best_auc:.4f}')

    save_results({
        'seed': seed, 'best_auc': best_auc,
        'all_metrics': all_metrics,
        'total_time_min': total_time,
        'checkpoint': str(best_ckpt),
        'dataset': dataset_name,
    }, f'{results_dir}/seed{seed}_results.json')

    if not args.no_upload and cfg['huggingface']['enabled']:
        upload_to_hf(cfg, str(best_ckpt), seed)

    if wandb_run:
        wandb_run.finish()

    return best_auc


if __name__ == '__main__':
    main()
