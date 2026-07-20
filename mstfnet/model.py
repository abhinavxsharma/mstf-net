"""
mstfnet/model.py
================
MSTF-Net: complete model assembling all three streams,
Laplacian quality estimator, DSTG fusion, and classifier.

Quick usage:
    from mstfnet import MSTFNet
    model = MSTFNet()
    logits, alpha, q_n = model(frames)   # frames: (B, 3, 224, 224)

Authors : Abhinav Vats et al., Chandigarh University
Accepted: ISSMAD 2026 (Google + IEEE Signal Processing Society)
"""

import torch
import torch.nn as nn

from .streams import SpatialStream, SpectralStream, SRMStream
from .fusion  import LaplacianQualityEstimator, DSTGFusion


class MSTFNet(nn.Module):
    """
    MSTF-Net v3 — full three-stream model with DSTG quality-aware fusion.

    Architecture:
        x ──► SpatialStream  (EfficientNet-B0)    ──► f_s (B,256)  ─┐
        x ──► SpectralStream (ResNet-18+FAA+DCT)  ──► f_f (B,256)  ─┼──► DSTG ──► Classifier
        x ──► SRMStream      (fixed SRM kernels)  ──► f_n (B,256)  ─┘
        x ──► LaplacianQE                         ──► Q_n (B,)     ─┘

    Args:
        d          : feature dimension (default 256)
        tau_min    : minimum DSTG temperature (default 0.1)
        tau_max    : maximum DSTG temperature (default 2.0)
        cls_dropout: dropout before classifier (default 0.4)

    Forward returns:
        logits : (B, 2)  — real/fake classification logits
        alpha  : (B, 3)  — per-stream fusion weights [α_s, α_f, α_n]
        q_n    : (B,)    — per-frame quality score

    Results (paper):
        DeeperForensics-1.0 : AUC 0.9792 ± 0.0025 (5 seeds)
        FF++ C23            : AUC 0.7269 ± 0.0101 (3 seeds)
        FF++ C40            : AUC 0.7143 ± 0.0103 (3× robustness gain)
        Total params        : 29.6M
        Inference           : 11.2 ms/frame on H200
    """

    def __init__(
        self,
        d: int           = 256,
        tau_min: float   = 0.1,
        tau_max: float   = 2.0,
        cls_dropout: float = 0.4,
    ):
        super().__init__()

        # Three parallel feature streams
        self.spatial  = SpatialStream(d=d)
        self.spectral = SpectralStream(d=d)
        self.srm      = SRMStream(d=d)

        # Differentiable quality estimator
        self.quality  = LaplacianQualityEstimator()

        # DSTG adaptive fusion
        self.dstg     = DSTGFusion(d=d, tau_min=tau_min, tau_max=tau_max)

        # Two-layer MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(d, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, 3, 224, 224)  normalised RGB frames

        Returns:
            logits : (B, 2)
            alpha  : (B, 3)  fusion weights — useful for visualisation
            q_n    : (B,)    quality scores — useful for analysis
        """
        # Parallel stream forward passes
        f_s = self.spatial(x)    # (B, 256)
        f_f = self.spectral(x)   # (B, 256)
        f_n = self.srm(x)        # (B, 256)

        # Quality estimation
        q_n = self.quality(x)    # (B,)

        # DSTG fusion
        fused, alpha = self.dstg([f_s, f_f, f_n], q_n)   # (B,256), (B,3)

        # Classification
        logits = self.classifier(fused)                    # (B, 2)

        return logits, alpha, q_n

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method — returns fake probability ∈ [0, 1].
        Args:
            x : (B, 3, 224, 224)
        Returns:
            probs : (B,)  — P(fake)
        """
        with torch.no_grad():
            logits, _, _ = self.forward(x)
            return torch.softmax(logits, dim=1)[:, 1]

    def count_parameters(self) -> dict:
        """Returns parameter count breakdown per component."""
        def n(m): return sum(p.numel() for p in m.parameters()) / 1e6
        return {
            'spatial_stream'  : f'{n(self.spatial):.2f}M',
            'spectral_stream' : f'{n(self.spectral):.2f}M',
            'srm_stream'      : f'{n(self.srm):.2f}M',
            'quality_estimator': f'{n(self.quality):.6f}M',
            'dstg_fusion'     : f'{n(self.dstg):.2f}M',
            'classifier'      : f'{n(self.classifier):.4f}M',
            'total'           : f'{n(self):.2f}M',
        }


# ─────────────────────────────────────────────────────────────────────────────
# Ablation variant — configurable streams for reproducing Table 4 in paper
# ─────────────────────────────────────────────────────────────────────────────

class MSTFNetAblation(nn.Module):
    """
    Ablation variant of MSTF-Net. Allows disabling individual streams,
    switching to static fusion, or disabling quality conditioning.

    Used to reproduce all 8 configurations in Table 4 of the paper.

    Args:
        streams     : list of stream indices to use, e.g. [1,2,3] or [1,3]
                      1=Spatial, 2=Spectral, 3=SRM
        use_dstg    : if False, uses simple mean pooling instead of DSTG
        use_quality : if False, sets Q_n ≡ 1 (no compression-awareness)
        d           : feature dimension (256)
    """

    def __init__(
        self,
        streams: list      = [1, 2, 3],
        use_dstg: bool     = True,
        use_quality: bool  = True,
        d: int             = 256,
        tau_min: float     = 0.1,
        tau_max: float     = 2.0,
    ):
        super().__init__()
        self.stream_ids  = streams
        self.use_dstg    = use_dstg
        self.use_quality = use_quality

        if 1 in streams: self.spatial  = SpatialStream(d=d)
        if 2 in streams: self.spectral = SpectralStream(d=d)
        if 3 in streams: self.srm      = SRMStream(d=d)
        if use_quality:  self.quality  = LaplacianQualityEstimator()

        self.fusion = (
            DSTGFusion(d=d, n_streams=len(streams),
                       tau_min=tau_min, tau_max=tau_max)
            if (use_dstg and len(streams) == 3)
            else None
        )

        self.classifier = nn.Sequential(
            nn.Linear(d, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor):
        feats = []
        if 1 in self.stream_ids: feats.append(self.spatial(x))
        if 2 in self.stream_ids: feats.append(self.spectral(x))
        if 3 in self.stream_ids: feats.append(self.srm(x))

        if self.fusion is not None:
            q_n = (self.quality(x) if self.use_quality
                   else torch.ones(x.size(0), device=x.device))
            fused, alpha = self.fusion(feats, q_n)
        else:
            # Static fusion — simple mean
            fused = torch.stack(feats, dim=0).mean(dim=0)
            alpha = None

        return self.classifier(fused), alpha
