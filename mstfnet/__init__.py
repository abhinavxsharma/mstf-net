"""
mstfnet — Adaptive Multi-Stream Deepfake Detection
===================================================
MSTF-Net: Adaptive Multi-Stream Deepfake Detection via
Dynamic Spectral-Temporal Gating

Accepted at ISSMAD 2026
(co-sponsored by Google + IEEE Signal Processing Society)

Authors: Abhinav Vats, Poonam Jyoti, Ishika Bhardwaj,
         Tanvi Garg, Tannu Ghanghas
         Chandigarh University, Mohali, Punjab, India

Quick start:
    from mstfnet import MSTFNet
    import torch

    model = MSTFNet()
    x = torch.randn(4, 3, 224, 224)
    logits, alpha, q_n = model(x)
    probs = model.predict_proba(x)
"""

from .model   import MSTFNet, MSTFNetAblation
from .streams import SpatialStream, SpectralStream, SRMStream
from .fusion  import LaplacianQualityEstimator, DSTGFusion
from .dataset import (
    WildDeepfakeDataset,
    VideoDataset,
    get_train_transform,
    get_val_transform,
)
from .utils import (
    set_seed,
    compute_metrics,
    save_checkpoint,
    load_checkpoint,
    save_results,
    load_results,
    get_class_weights,
    AverageMeter,
    EarlyStopping,
    print_gpu_info,
)

__version__ = '1.0.0'
__author__  = 'Abhinav Vats et al.'
__paper__   = ('MSTF-Net: Adaptive Multi-Stream Deepfake Detection'
               ' via Dynamic Spectral-Temporal Gating')
__venue__   = 'ISSMAD 2026 (Google + IEEE Signal Processing Society)'

__all__ = [
    'MSTFNet', 'MSTFNetAblation',
    'SpatialStream', 'SpectralStream', 'SRMStream',
    'LaplacianQualityEstimator', 'DSTGFusion',
    'WildDeepfakeDataset', 'VideoDataset',
    'get_train_transform', 'get_val_transform',
    'set_seed', 'compute_metrics',
    'save_checkpoint', 'load_checkpoint',
    'save_results', 'load_results',
    'get_class_weights', 'AverageMeter', 'EarlyStopping',
    'print_gpu_info',
]