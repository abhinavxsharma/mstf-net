"""
mstfnet/fusion.py
=================
Dynamic Spectral-Temporal Gating (DSTG) and Laplacian Quality Estimator.

Key idea — reliability-induced fusion collapse:
  Static fusion weights degraded streams (spectral, SRM) equally with the
  robust spatial stream under compression, causing AUC collapse. DSTG
  conditions per-stream weights on a differentiable quality score Q_n,
  suppressing unreliable streams at low quality (C40) and distributing
  weight evenly at high quality (C23).

Authors : Abhinav Vats et al., Chandigarh University
Paper   : MSTF-Net: Adaptive Multi-Stream Deepfake Detection
Accepted: ISSMAD 2026 (Google + IEEE Signal Processing Society)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Laplacian Quality Estimator
# ─────────────────────────────────────────────────────────────────────────────

class LaplacianQualityEstimator(nn.Module):
    """
    Differentiable per-frame compression quality score Q_n ∈ (0, 1).

    Q_n(x) = σ( e^α · log Var[∇²_L * g(x)] + β )

    where g(x) is the luminance channel, ∇²_L is the 3×3 discrete Laplacian,
    and α, β are learnable scalars initialised to 0.

    Calibrated values (from paper):
      C23 frames → Q_n ≈ 0.78  (high quality)
      C40 frames → Q_n ≈ 0.33  (heavily compressed)

    No quality supervision required — α and β are learned end-to-end.

    Input  : (B, 3, 224, 224)
    Output : (B,)  — scalar quality score per frame
    Params : 2 (α and β only)
    """

    def __init__(self):
        super().__init__()
        # Fixed 3×3 Laplacian kernel
        lap = torch.tensor([
            [ 0.,  1.,  0.],
            [ 1., -4.,  1.],
            [ 0.,  1.,  0.],
        ])
        self.register_buffer('laplacian', lap.view(1, 1, 3, 3))
        # Learnable scalars — initialised to 0
        self.alpha = nn.Parameter(torch.tensor(0.))
        self.beta  = nn.Parameter(torch.tensor(0.))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RGB → luminance  (BT.601)
        g = (0.299 * x[:, 0]
           + 0.587 * x[:, 1]
           + 0.114 * x[:, 2]).unsqueeze(1)         # (B, 1, H, W)

        # Apply Laplacian filter
        lap_response = F.conv2d(g, self.laplacian, padding=1)  # (B,1,H,W)

        # Log variance of Laplacian response
        var = lap_response.view(x.size(0), -1).var(dim=1).clamp(min=1e-6)
        log_var = torch.log(var)                    # (B,)

        # Learnable affine + sigmoid
        q = torch.sigmoid(self.alpha.exp() * log_var + self.beta)
        return q                                    # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Gate MLP  (one per stream)
# ─────────────────────────────────────────────────────────────────────────────

def _make_gate_mlp() -> nn.Module:
    """
    Gate MLP: Linear(1→32)→GELU→Linear(32→16)→GELU→Linear(16→1)
    Maps Q_n scalar → unnormalised gate logit g_i
    """
    return nn.Sequential(
        nn.Linear(1, 32),
        nn.GELU(),
        nn.Linear(32, 16),
        nn.GELU(),
        nn.Linear(16, 1),
    )


def _make_temp_mlp() -> nn.Module:
    """
    Temperature MLP: Linear(1→16)→GELU→Linear(16→1)→Sigmoid
    Maps Q_n → temperature τ_i ∈ (τ_min, τ_max)
    """
    return nn.Sequential(
        nn.Linear(1, 16),
        nn.GELU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DSTG Fusion Module
# ─────────────────────────────────────────────────────────────────────────────

class DSTGFusion(nn.Module):
    """
    Dynamic Spectral-Temporal Gating (DSTG).

    For each stream i ∈ {1,2,3}:
        g_i   = G_i(Q_n)
        τ_i   = τ_min + (τ_max - τ_min) · T_i(Q_n)
        α     = softmax(g_1/τ_1, g_2/τ_2, g_3/τ_3)

    Weighted sum + 4-head cross-stream attention:
        f_fused = LN(GELU(W_o · LN(Ā + Σ α_i f_i)))

    where Ā is mean-pooled multi-head attention output.

    Behaviour:
      Low Q_n  (C40, 0.33) → high τ → sharper softmax → spatial dominates
      High Q_n (C23, 0.78) → low τ  → even distribution across all 3 streams

    Input  : list of 3 tensors each (B, 256), scalar Q_n (B,)
    Output : (B, 256) fused feature, (B, 3) gating weights α
    Params : ~0.3M
    """

    def __init__(
        self,
        d: int   = 256,
        n_streams: int = 3,
        tau_min: float = 0.1,
        tau_max: float = 2.0,
        n_heads: int   = 4,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Independent gate + temperature MLP per stream
        self.gate_mlps = nn.ModuleList(
            [_make_gate_mlp() for _ in range(n_streams)]
        )
        self.temp_mlps = nn.ModuleList(
            [_make_temp_mlp() for _ in range(n_streams)]
        )

        # 4-head cross-stream self-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.ln_attn = nn.LayerNorm(d)

        # Output projection
        self.out = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

    def forward(
        self,
        streams: list,         # [f_s, f_f, f_n] each (B, 256)
        q_n: torch.Tensor,     # (B,)
    ):
        q = q_n.unsqueeze(1)   # (B, 1)

        # Compute per-stream gate logits and temperatures
        gates = [m(q).squeeze(1) for m in self.gate_mlps]   # [(B,), ...]
        temps = [
            self.tau_min + (self.tau_max - self.tau_min) * m(q).squeeze(1)
            for m in self.temp_mlps
        ]                                                     # [(B,), ...]

        # Temperature-scaled softmax → α ∈ simplex
        logits = torch.stack(
            [g / t.clamp(min=1e-3) for g, t in zip(gates, temps)],
            dim=1
        )                                                     # (B, 3)
        alpha = F.softmax(logits, dim=1)                      # (B, 3)

        # Weighted sum of stream features
        stack  = torch.stack(streams, dim=1)                  # (B, 3, 256)
        fused  = (stack * alpha.unsqueeze(-1)).sum(dim=1)     # (B, 256)

        # 4-head cross-stream attention
        attn_out, _ = self.cross_attn(stack, stack, stack)   # (B, 3, 256)
        attn_mean   = attn_out.mean(dim=1)                    # (B, 256)

        # Residual combination + output projection
        combined = self.ln_attn(fused + attn_mean)            # (B, 256)
        return self.out(combined), alpha                       # (B,256), (B,3)
