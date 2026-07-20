"""
tests/test_model.py
===================
Unit tests for MSTF-Net model components.
Run: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np

from mstfnet import (
    MSTFNet, MSTFNetAblation,
    SpatialStream, SpectralStream, SRMStream,
    LaplacianQualityEstimator, DSTGFusion,
)


DEVICE = torch.device('cpu')
B = 2   # batch size for tests


@pytest.fixture
def dummy_input():
    return torch.randn(B, 3, 224, 224)


# ── Stream tests ─────────────────────────────────────────────

def test_spatial_stream_output_shape(dummy_input):
    model = SpatialStream(d=256)
    out   = model(dummy_input)
    assert out.shape == (B, 256), f'Expected (B,256) got {out.shape}'


def test_spectral_stream_output_shape(dummy_input):
    model = SpectralStream(d=256)
    out   = model(dummy_input)
    assert out.shape == (B, 256), f'Expected (B,256) got {out.shape}'


def test_srm_stream_output_shape(dummy_input):
    model = SRMStream(d=256)
    out   = model(dummy_input)
    assert out.shape == (B, 256), f'Expected (B,256) got {out.shape}'


def test_srm_kernels_not_learned(dummy_input):
    """SRM filter weights must be fixed buffers, not parameters."""
    model = SRMStream()
    param_names = [n for n, _ in model.named_parameters()]
    assert 'srm_kernels' not in param_names, \
        'srm_kernels should be a buffer, not a parameter'


def test_srm_kernel_count():
    """Must have exactly 3 SRM kernels of size 5×5."""
    model = SRMStream()
    assert model.srm_kernels.shape == (3, 1, 5, 5), \
        f'Expected (3,1,5,5) got {model.srm_kernels.shape}'


# ── Quality Estimator tests ───────────────────────────────────

def test_quality_estimator_range(dummy_input):
    """Q_n must be in (0, 1) for all inputs."""
    model = LaplacianQualityEstimator()
    q_n   = model(dummy_input)
    assert q_n.shape == (B,)
    assert (q_n > 0).all() and (q_n < 1).all(), \
        f'Q_n out of (0,1): min={q_n.min():.4f} max={q_n.max():.4f}'


def test_quality_estimator_only_2_params():
    """Quality estimator must have exactly 2 learnable params (alpha, beta)."""
    model  = LaplacianQualityEstimator()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 2, f'Expected 2 params, got {n_params}'


# ── DSTG Fusion tests ─────────────────────────────────────────

def test_dstg_output_shape():
    model   = DSTGFusion(d=256)
    streams = [torch.randn(B, 256) for _ in range(3)]
    q_n     = torch.rand(B)
    fused, alpha = model(streams, q_n)
    assert fused.shape == (B, 256)
    assert alpha.shape == (B, 3)


def test_dstg_alpha_sums_to_one():
    """Gating weights must form a valid probability simplex."""
    model   = DSTGFusion(d=256)
    streams = [torch.randn(B, 256) for _ in range(3)]
    q_n     = torch.rand(B)
    _, alpha = model(streams, q_n)
    sums = alpha.sum(dim=1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), \
        f'Alpha does not sum to 1: {sums}'


# ── Full Model tests ──────────────────────────────────────────

def test_mstfnet_output_shapes(dummy_input):
    model = MSTFNet()
    logits, alpha, q_n = model(dummy_input)
    assert logits.shape == (B, 2)
    assert alpha.shape  == (B, 3)
    assert q_n.shape    == (B,)


def test_mstfnet_no_nan(dummy_input):
    model  = MSTFNet()
    logits, alpha, q_n = model(dummy_input)
    assert not torch.isnan(logits).any(), 'NaN in logits'
    assert not torch.isnan(alpha).any(),  'NaN in alpha'
    assert not torch.isnan(q_n).any(),    'NaN in q_n'


def test_mstfnet_predict_proba(dummy_input):
    model = MSTFNet()
    probs = model.predict_proba(dummy_input)
    assert probs.shape == (B,)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_mstfnet_param_count():
    """Total params should be ~29.6M."""
    model    = MSTFNet()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    assert 25 < n_params < 35, \
        f'Unexpected param count: {n_params:.1f}M (expected ~29.6M)'


# ── Ablation variant tests ────────────────────────────────────

@pytest.mark.parametrize('streams,use_dstg,use_quality', [
    ([1],       False, False),   # Spatial only
    ([3],       False, False),   # SRM only
    ([2],       False, False),   # Spectral only
    ([1, 2],    False, False),   # Spatial + Spectral
    ([1, 3],    False, False),   # Spatial + SRM
    ([1, 2, 3], False, False),   # All streams static
    ([1, 2, 3], True,  False),   # DSTG no quality
    ([1, 2, 3], True,  True),    # Full MSTF-Net
])
def test_ablation_configs(dummy_input, streams, use_dstg, use_quality):
    model  = MSTFNetAblation(
        streams=streams, use_dstg=use_dstg, use_quality=use_quality
    )
    out, alpha = model(dummy_input)
    assert out.shape == (B, 2), \
        f'streams={streams} dstg={use_dstg} q={use_quality}: shape={out.shape}'
    assert not torch.isnan(out).any(), \
        f'NaN in output for streams={streams}'
