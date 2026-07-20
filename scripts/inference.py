"""
scripts/inference.py
====================
Run MSTF-Net inference on any video file.
Outputs: REAL/FAKE label, confidence score, per-frame breakdown.

Usage:
    python scripts/inference.py \
        --video path/to/video.mp4 \
        --checkpoint path/to/mstfnet_best.pth

    # Multiple videos
    python scripts/inference.py \
        --video_dir path/to/videos/ \
        --checkpoint path/to/mstfnet_best.pth

Authors : Abhinav Vats et al., Chandigarh University
Accepted: ISSMAD 2026 (Google + IEEE Signal Processing Society)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from mstfnet import MSTFNet, VideoDataset, load_checkpoint, print_gpu_info


def parse_args():
    p = argparse.ArgumentParser(description='MSTF-Net Video Inference')
    p.add_argument('--video',      type=str, default=None,
                   help='Path to single video file')
    p.add_argument('--video_dir',  type=str, default=None,
                   help='Directory of videos to process')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to .pth checkpoint')
    p.add_argument('--n_frames',   type=int, default=8)
    p.add_argument('--threshold',  type=float, default=0.5,
                   help='Decision threshold for FAKE classification')
    p.add_argument('--device',     type=str, default='auto')
    return p.parse_args()


@torch.no_grad()
def predict_video(model, video_path, n_frames, device, threshold=0.5):
    """
    Run MSTF-Net on a single video.

    Returns dict with:
        label      : 'FAKE' or 'REAL'
        confidence : float 0-1 (fake probability)
        per_frame  : list of per-frame fake probabilities
        mean_qn    : mean Laplacian quality score
        mean_alpha : mean DSTG gating weights [spatial, spectral, srm]
    """
    ds     = VideoDataset([str(video_path)], n_frames=n_frames)
    frames, _ = ds[0]                        # (n_frames, 3, H, W)

    if frames.sum() == 0:
        return {'label': 'ERROR', 'confidence': 0.0,
                'error': 'Could not read video'}

    frames = frames.to(device)               # (n_frames, 3, H, W)

    # Process frame by frame
    per_frame_probs  = []
    per_frame_alpha  = []
    per_frame_qn     = []

    model.eval()
    for i in range(frames.size(0)):
        frame = frames[i].unsqueeze(0)       # (1, 3, H, W)
        logits, alpha, q_n = model(frame)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        per_frame_probs.append(prob)
        per_frame_alpha.append(alpha[0].cpu().numpy().tolist())
        per_frame_qn.append(q_n[0].item())

    mean_prob  = float(np.mean(per_frame_probs))
    label      = 'FAKE' if mean_prob >= threshold else 'REAL'
    mean_alpha = np.mean(per_frame_alpha, axis=0).tolist()
    mean_qn    = float(np.mean(per_frame_qn))

    return {
        'video'      : str(video_path),
        'label'      : label,
        'confidence' : mean_prob,
        'per_frame'  : per_frame_probs,
        'mean_qn'    : mean_qn,
        'mean_alpha' : {
            'spatial'  : mean_alpha[0],
            'spectral' : mean_alpha[1],
            'srm'      : mean_alpha[2],
        },
        'threshold'  : threshold,
    }


def print_result(result):
    """Pretty-print a single video result."""
    label = result['label']
    conf  = result['confidence']
    bar   = '█' * int(conf * 30) + '░' * (30 - int(conf * 30))

    print(f'\n{"="*55}')
    print(f'Video     : {Path(result["video"]).name}')
    print(f'Prediction: {label}  ({conf*100:.1f}% fake)')
    print(f'Confidence: [{bar}]')
    print(f'Quality Qn: {result["mean_qn"]:.3f}  '
          f'(0.78=C23 quality, 0.33=C40 quality)')
    print(f'DSTG weights:')
    alpha = result['mean_alpha']
    print(f'  Spatial  : {alpha["spatial"]:.3f}')
    print(f'  Spectral : {alpha["spectral"]:.3f}')
    print(f'  SRM      : {alpha["srm"]:.3f}')
    print(f'Per-frame  : {[f"{p:.3f}" for p in result["per_frame"]]}')
    print(f'{"="*55}')


def main():
    args   = parse_args()
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    if args.device != 'auto':
        device = torch.device(args.device)

    print_gpu_info()

    # Load model
    print(f'\nLoading checkpoint: {args.checkpoint}')
    model = MSTFNet().to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    # Collect video paths
    if args.video:
        video_paths = [Path(args.video)]
    elif args.video_dir:
        video_dir   = Path(args.video_dir)
        video_paths = sorted(
            list(video_dir.glob('*.mp4')) +
            list(video_dir.glob('*.avi')) +
            list(video_dir.glob('*.mov'))
        )
        print(f'Found {len(video_paths)} videos in {video_dir}')
    else:
        print('Provide --video or --video_dir')
        return

    # Run inference
    all_results = []
    for vpath in video_paths:
        print(f'\nProcessing: {vpath.name}')
        result = predict_video(
            model, vpath, args.n_frames, device, args.threshold
        )
        print_result(result)
        all_results.append(result)

    # Summary if multiple videos
    if len(all_results) > 1:
        n_fake = sum(1 for r in all_results if r['label'] == 'FAKE')
        n_real = len(all_results) - n_fake
        print(f'\nSUMMARY: {n_fake} FAKE / {n_real} REAL '
              f'out of {len(all_results)} videos')


if __name__ == '__main__':
    main()
