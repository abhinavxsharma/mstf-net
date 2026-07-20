"""
mstfnet/dataset.py
==================
Dataset classes for MSTF-Net — WildDeepfake edition.

WildDeepfake structure (after extraction):
    wilddeepfake/
    ├── train/
    │   ├── real/   {sequence_id}/  *.png
    │   └── fake/   {sequence_id}/  *.png
    └── test/
        ├── real/   {sequence_id}/  *.png
        └── fake/   {sequence_id}/  *.png

Key facts:
  - Images are already PNG face crops (no video extraction needed)
  - Each folder = one face sequence = multiple frames
  - We sample up to N_FRAMES uniformly from each sequence
  - Official train/test split provided
  - ~3805 real + ~3509 fake sequences total
  - Roughly balanced (~50/50) — WeightedRandomSampler still used for safety

Protocol matches paper:
  - 8 frames per sequence (uniformly sampled)
  - Resize 224×224
  - ImageNet normalisation
  - Train: RandomHorizontalFlip + RandomRotation(10)
  - Val/Test: No augmentation

Authors : Abhinav Vats et al., Chandigarh University
Accepted: ISSMAD 2026 (Google + IEEE Signal Processing Society)
"""

import random
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Transforms ────────────────────────────────────────────────

def get_train_transform(img_size: int = 224) -> transforms.Compose:
    """Exactly matches paper protocol — no ColorJitter."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_val_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── WildDeepfake Dataset ───────────────────────────────────────

class WildDeepfakeDataset(Dataset):
    """
    WildDeepfake dataset loader.

    Loads PNG face frames from extracted sequence folders.
    Samples up to n_frames uniformly per sequence.

    Args:
        root      : path to split root (e.g. wilddeepfake/train)
        split     : 'train' or 'test'
        transform : torchvision transform
        n_frames  : max frames to sample per sequence (default 8)

    Returns per item:
        (img_tensor, label, source)
        label  : 0=real, 1=fake
        source : 'real' or 'fake'
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
        n_frames: int = 8,
    ):
        assert split in ('train', 'test')
        self.transform = transform or get_val_transform()
        self.n_frames  = n_frames
        self.samples: List[Tuple[str, int, str]] = []

        base = Path(root) / split

        # Real sequences → label 0
        real_dir = base / 'real'
        if real_dir.exists():
            for seq_dir in self._find_sequence_dirs(real_dir):
                frames = sorted(seq_dir.glob('*.png'))
                if not frames:
                    frames = sorted(seq_dir.glob('*.jpg'))
                if frames:
                    for f in self._sample_frames(frames):
                        self.samples.append((str(f), 0, 'real'))

        # Fake sequences → label 1
        fake_dir = base / 'fake'
        if fake_dir.exists():
            for seq_dir in self._find_sequence_dirs(fake_dir):
                frames = sorted(seq_dir.glob('*.png'))
                if not frames:
                    frames = sorted(seq_dir.glob('*.jpg'))
                if frames:
                    for f in self._sample_frames(frames):
                        self.samples.append((str(f), 1, 'fake'))

        random.shuffle(self.samples)

        n_real = sum(1 for _, l, _ in self.samples if l == 0)
        n_fake = sum(1 for _, l, _ in self.samples if l == 1)
        print(f'  WildDeepfake [{split}]: '
              f'real={n_real:,}  fake={n_fake:,}  '
              f'total={len(self.samples):,}')

    @staticmethod
    def _find_sequence_dirs(root_dir: Path) -> list:
        """
        Find every leaf directory containing images, at any nesting depth.

        WildDeepfake's raw archives don't always extract to a flat
        `<label>/<sequence_id>/*.png` layout — some per-video tars wrap
        their sequences in an extra label folder, e.g.
        `<label>/<video_id>/<label>/<sequence_id>/*.png`. Rather than
        assume a fixed depth, walk the tree and treat any directory
        that directly contains .png/.jpg files (and has no subdirectories
        of its own containing images) as one sequence.
        """
        import os
        seq_dirs = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            has_images = any(
                fn.lower().endswith(('.png', '.jpg', '.jpeg'))
                for fn in filenames
            )
            if has_images:
                seq_dirs.append(Path(dirpath))
        return sorted(seq_dirs)

    def _sample_frames(self, frames: list) -> list:
        """Uniformly sample up to n_frames from a sequence."""
        n = min(self.n_frames, len(frames))
        if n == len(frames):
            return frames
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        return [frames[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label, source = self.samples[i]
        with open(path, 'rb') as f:
            data = f.read()
        img = Image.open(BytesIO(data)).convert('RGB')
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long), source

    @property
    def labels(self) -> List[int]:
        return [l for _, l, _ in self.samples]


# ── Video inference dataset (unchanged) ───────────────────────

class VideoDataset(Dataset):
    """On-the-fly frame extraction for inference on raw video files."""

    def __init__(self, video_paths: List[str], n_frames: int = 8,
                 img_size: int = 224):
        import cv2
        self.video_paths = [str(p) for p in video_paths]
        self.n_frames    = n_frames
        self.transform   = get_val_transform(img_size)
        self._cv2        = cv2

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int):
        path  = self.video_paths[idx]
        cap   = self._cv2.VideoCapture(path)
        total = int(cap.get(self._cv2.CAP_PROP_FRAME_COUNT))
        idxs  = np.linspace(0, max(total-1,0),
                             min(self.n_frames, max(total,1)), dtype=int)
        frames = []
        for i in idxs:
            cap.set(self._cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok:
                frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
                frame = self._cv2.resize(frame, (224, 224))
                frames.append(self.transform(Image.fromarray(frame)))
        cap.release()
        if not frames:
            return torch.zeros(self.n_frames, 3, 224, 224), path
        return torch.stack(frames), path