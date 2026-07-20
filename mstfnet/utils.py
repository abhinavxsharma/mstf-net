"""
mstfnet/utils.py
================
Shared utilities: reproducibility, metrics, checkpointing, logging.

Authors : Abhinav Vats et al., Chandigarh University
Accepted: ISSMAD 2026 (Google + IEEE Signal Processing Society)
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute AUC, accuracy, and F1 from ground-truth labels and fake probabilities.

    Args:
        y_true    : (N,) integer labels  0=real, 1=fake
        y_prob    : (N,) fake probability scores
        threshold : decision threshold for accuracy/F1

    Returns:
        dict with keys: auc, accuracy, f1
    """
    if len(np.unique(y_true)) < 2:
        return {'auc': 0.5, 'accuracy': 0.0, 'f1': 0.0}

    auc  = float(roc_auc_score(y_true, y_prob))
    pred = (y_prob >= threshold).astype(int)
    acc  = float(accuracy_score(y_true, pred))
    f1   = float(f1_score(y_true, pred, zero_division=0))

    return {'auc': auc, 'accuracy': acc, 'f1': f1}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    metrics: dict,
    epoch: int,
    seed: int,
    save_dir: str,
    filename: Optional[str] = None,
) -> str:
    """
    Save model checkpoint with metadata.

    Args:
        model    : PyTorch model
        metrics  : dict of metrics at this checkpoint
        epoch    : current epoch number
        seed     : training seed
        save_dir : directory to save checkpoint
        filename : optional override for filename

    Returns:
        path to saved checkpoint
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fname = filename or f'mstfnet_seed{seed}_epoch{epoch:02d}.pth'
    path  = str(Path(save_dir) / fname)

    torch.save({
        'epoch'       : epoch,
        'seed'        : seed,
        'metrics'     : metrics,
        'model_state' : model.state_dict(),
        'timestamp'   : time.strftime('%Y-%m-%d %H:%M:%S'),
    }, path)

    return path


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
) -> dict:
    """
    Load a saved checkpoint into model.

    Returns:
        checkpoint dict (contains epoch, seed, metrics, timestamp)
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'], strict=strict)
    print(f'Loaded checkpoint: {checkpoint_path}')
    print(f'  Epoch   : {ckpt.get("epoch", "?")}')
    print(f'  Seed    : {ckpt.get("seed", "?")}')
    print(f'  Metrics : {ckpt.get("metrics", {})}')
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Results I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, path: str) -> None:
    """Save results dict as pretty-printed JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _convert(o):
        if isinstance(o, (np.floating, float)): return float(o)
        if isinstance(o, (np.integer, int)):    return int(o)
        if isinstance(o, np.ndarray):           return o.tolist()
        return o

    with open(path, 'w') as f:
        json.dump(results, f, default=_convert, indent=2)
    print(f'Results saved → {path}')


def load_results(path: str) -> dict:
    """Load results JSON."""
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_class_weights(labels: List[int]) -> List[float]:
    """
    Compute per-sample weights for WeightedRandomSampler.
    Balances class imbalance (e.g. 1:4 real:fake ratio).

    Args:
        labels : list of integer class labels

    Returns:
        list of per-sample weights
    """
    counts  = {l: labels.count(l) for l in set(labels)}
    weights = [1.0 / counts[l] for l in labels]
    return weights


class AverageMeter:
    """Tracks running average of a scalar (e.g. loss)."""

    def __init__(self, name: str = ''):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.
        self.avg   = 0.
        self.sum   = 0.
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'


class EarlyStopping:
    """
    Early stopping based on validation AUC.

    Args:
        patience  : epochs to wait after last improvement (default 4)
        min_delta : minimum improvement to count as improvement
    """

    def __init__(self, patience: int = 4, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = -float('inf')
        self.counter    = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """
        Call after each epoch.
        Returns True if training should stop.
        """
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ─────────────────────────────────────────────────────────────────────────────
# GPU info
# ─────────────────────────────────────────────────────────────────────────────

def print_gpu_info() -> None:
    """Print GPU name and VRAM."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'GPU  : {name}')
        print(f'VRAM : {vram:.1f} GB')
    else:
        print('No GPU available — running on CPU')
