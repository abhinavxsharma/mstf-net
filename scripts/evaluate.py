"""
scripts/evaluate.py
===================
Evaluate MSTF-Net checkpoints and aggregate results across seeds.

Usage:
    # Evaluate single checkpoint
    python scripts/evaluate.py \
        --checkpoint /path/to/mstfnet_best.pth \
        --frames_root /teamspace/studios/this_studio/DF_Frames

    # Aggregate all seeds into final_results.json
    python scripts/evaluate.py \
        --checkpoint_dir /teamspace/studios/this_studio/MSTF_checkpoints \
        --frames_root /teamspace/studios/this_studio/DF_Frames \
        --results_dir /teamspace/studios/this_studio/MSTF_results \
        --aggregate

Authors : Abhinav Vats et al., Chandigarh University
Accepted: ISSMAD 2026 (Google + IEEE Signal Processing Society)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from mstfnet import (
    MSTFNet,
    FrameDataset,
    get_val_transform,
    load_checkpoint,
    compute_metrics,
    save_results,
    print_gpu_info,
)


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate MSTF-Net')
    p.add_argument('--checkpoint',     type=str, default=None)
    p.add_argument('--checkpoint_dir', type=str, default=None)
    p.add_argument('--frames_root',    type=str,
                   default='/teamspace/studios/this_studio/DF_Frames')
    p.add_argument('--results_dir',    type=str,
                   default='/teamspace/studios/this_studio/MSTF_results')
    p.add_argument('--batch_size',     type=int, default=256)
    p.add_argument('--aggregate',      action='store_true')
    return p.parse_args()


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path, frames_root, batch_size, device):
    """Evaluate a single checkpoint on the val set."""
    model = MSTFNet().to(device)
    ckpt  = load_checkpoint(model, checkpoint_path, device)

    val_ds = FrameDataset(
        root=f'{frames_root}/val',
        transform=get_val_transform(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True,
    )

    model.eval()
    all_probs, all_labels = [], []

    for imgs, labels in val_loader:
        imgs  = imgs.to(device, non_blocking=True)
        probs = model.predict_proba(imgs).cpu().numpy()
        all_probs.extend(np.nan_to_num(probs, nan=0.5))
        all_labels.extend(labels.numpy())
        del imgs

    y_true   = np.array(all_labels, dtype=np.int32)
    y_prob   = np.array(all_probs,  dtype=np.float32)
    metrics  = compute_metrics(y_true, y_prob)
    metrics['seed']       = ckpt.get('seed', '?')
    metrics['epoch']      = ckpt.get('epoch', '?')
    metrics['checkpoint'] = checkpoint_path
    return metrics


def aggregate_seeds(checkpoint_dir, frames_root, results_dir, batch_size, device):
    """Find all BEST checkpoints, evaluate each, compute mean±std."""
    ckpt_dir = Path(checkpoint_dir)
    best_ckpts = sorted(ckpt_dir.glob('*_BEST.pth'))

    if not best_ckpts:
        print(f'No *_BEST.pth found in {checkpoint_dir}')
        return

    print(f'Found {len(best_ckpts)} best checkpoints:')
    for c in best_ckpts:
        print(f'  {c.name}')

    all_results = []
    for ckpt_path in best_ckpts:
        print(f'\nEvaluating {ckpt_path.name}...')
        metrics = evaluate_checkpoint(
            str(ckpt_path), frames_root, batch_size, device
        )
        all_results.append(metrics)
        print(f'  AUC={metrics["auc"]:.4f}  '
              f'Acc={metrics["accuracy"]:.4f}  '
              f'F1={metrics["f1"]:.4f}')

    # Compute mean ± std
    aucs = [r['auc'] for r in all_results]
    accs = [r['accuracy'] for r in all_results]
    f1s  = [r['f1'] for r in all_results]

    summary = {
        'mean_auc' : float(np.mean(aucs)),
        'std_auc'  : float(np.std(aucs)),
        'mean_acc' : float(np.mean(accs)),
        'std_acc'  : float(np.std(accs)),
        'mean_f1'  : float(np.mean(f1s)),
        'std_f1'   : float(np.std(f1s)),
        'all_aucs' : aucs,
        'formatted': f'{np.mean(aucs):.4f}±{np.std(aucs):.4f}',
        'per_seed' : all_results,
        'n_seeds'  : len(all_results),
    }

    print(f'\n{"="*50}')
    print(f'FINAL RESULT (DeeperForensics-1.0)')
    print(f'AUC : {summary["formatted"]}')
    print(f'Acc : {np.mean(accs):.4f}±{np.std(accs):.4f}')
    print(f'F1  : {np.mean(f1s):.4f}±{np.std(f1s):.4f}')
    print(f'{"="*50}')

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    save_results(summary, f'{results_dir}/final_results.json')


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_gpu_info()

    if args.aggregate:
        aggregate_seeds(
            args.checkpoint_dir, args.frames_root,
            args.results_dir, args.batch_size, device
        )
    elif args.checkpoint:
        metrics = evaluate_checkpoint(
            args.checkpoint, args.frames_root, args.batch_size, device
        )
        print(f'\nResults:')
        for k, v in metrics.items():
            print(f'  {k}: {v}')
    else:
        print('Provide --checkpoint or --checkpoint_dir --aggregate')


if __name__ == '__main__':
    main()
