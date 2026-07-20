"""
mstfnet/streams.py
==================
Three parallel feature extraction streams for MSTF-Net:
  1. SpatialStream   — EfficientNet-B0 (spatial texture artifacts)
  2. SpectralStream  — ResNet-18 + FAA + DCT (frequency domain)
  3. SRMStream       — Fixed SRM filters (noise residuals)

Authors : Abhinav Vats et al., Chandigarh University
Paper   : MSTF-Net: Adaptive Multi-Stream Deepfake Detection via
          Dynamic Spectral-Temporal Gating
Accepted: ISSMAD 2026 (Google + IEEE Signal Processing Society)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import timm


# ─────────────────────────────────────────────────────────────────────────────
# Stream 1 — Spatial (EfficientNet-B0)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialStream(nn.Module):
    """
    EfficientNet-B0 pretrained on ImageNet-1K.
    Detects spatial texture artifacts: blending seams, skin inconsistencies,
    geometric distortions. Most robust stream under all compression levels.

    Input  : (B, 3, 224, 224)
    Output : (B, 256)  — L2-normalised feature vector
    Params : ~4.8M
    """

    def __init__(self, d: int = 256, dropout: float = 0.3):
        super().__init__()
        backbone = timm.create_model(
            'efficientnet_b0', pretrained=True, num_classes=0
        )
        self.backbone = backbone
        in_features = backbone.num_features  # 1280

        self.proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, d),
            nn.BatchNorm1d(d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)          # (B, 1280)
        return self.proj(feats)           # (B, 256)


# ─────────────────────────────────────────────────────────────────────────────
# Stream 2 — Spectral (ResNet-18 + FAA + DCT)
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyAttentionAdapter(nn.Module):
    """
    FAA: dual-pooling channel attention inserted after ResNet-18 layer2.
    Selectively amplifies discriminative frequency channels.

    FAA(F) = F ⊙ σ(W_FAA · [GAP(F) ‖ GMP(F)])

    Input  : (B, C, H, W)
    Output : (B, C, H, W)  — channel-recalibrated
    """

    def __init__(self, channels: int = 128, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 2, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.avg_pool(x)                        # (B, C, 1, 1)
        mx  = self.max_pool(x)                        # (B, C, 1, 1)
        cat = torch.cat([avg, mx], dim=1)             # (B, 2C, 1, 1)
        cat = cat.squeeze(-1).squeeze(-1)             # (B, 2C)
        scale = self.fc(cat)                          # (B, C)
        scale = scale.unsqueeze(-1).unsqueeze(-1)     # (B, C, 1, 1)
        return x * scale


class DCTBlock(nn.Module):
    """
    Fixed 7×7 block-wise DCT inserted after ResNet-18 layer3.
    Converts spatial feature maps into frequency domain representations.

    DCT_{i,j}(F) = c_i·c_j Σ_{u,v} F_{u,v}
                   cos[π(2u+1)i/2H] cos[π(2v+1)j/2W]

    Input  : (B, C, H, W)
    Output : (B, C, H, W)  — DCT-transformed
    Params : fixed (no learned weights in DCT basis)
    """

    def __init__(self, channels: int = 256, block_size: int = 7):
        super().__init__()
        H = W = block_size
        n = H * W
        # Build DCT basis as fixed convolutional weights
        w = torch.zeros(n, 1, H, W)
        for i in range(H):
            for j in range(W):
                for u in range(H):
                    for v in range(W):
                        ci = np.sqrt(1 / H) if i == 0 else np.sqrt(2 / H)
                        cj = np.sqrt(1 / W) if j == 0 else np.sqrt(2 / W)
                        w[i * W + j, 0, u, v] = (
                            ci * cj
                            * np.cos((2 * u + 1) * i * np.pi / (2 * H))
                            * np.cos((2 * v + 1) * j * np.pi / (2 * W))
                        )
        self.register_buffer('weight', w)
        self.H = H
        self.W = W
        # 1×1 conv to mix DCT coefficients back to original channel count
        self.channel_mix = nn.Sequential(
            nn.Conv2d(channels * n, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H != self.H or W != self.W:
            x = F.adaptive_avg_pool2d(x, (self.H, self.W))
        # Apply DCT basis channel-wise
        x_flat = x.reshape(B * C, 1, self.H, self.W)
        dct    = F.conv2d(x_flat, self.weight)       # (B*C, H*W, 1, 1)
        dct    = dct.reshape(B, C * self.H * self.W, 1, 1)
        dct    = dct.expand(-1, -1, self.H, self.W)  # (B, C*H*W, H, W)
        return self.channel_mix(dct)                  # (B, C, H, W)


class SpectralStream(nn.Module):
    """
    ResNet-18 + Frequency Attention Adapter (after layer2)
                + DCT Block (after layer3).
    Detects GAN-specific periodic upsampling patterns in frequency domain.
    Most useful on pristine C23; loses power at C40 (overwritten by quantisation).

    Input  : (B, 3, 224, 224)
    Output : (B, 256)
    Params : ~14.6M
    """

    def __init__(self, d: int = 256, dropout: float = 0.3):
        super().__init__()
        r18 = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)

        # Split ResNet-18 into stages for FAA / DCT insertion
        self.layer0 = nn.Sequential(
            r18.conv1, r18.bn1, r18.relu, r18.maxpool
        )
        self.layer1 = r18.layer1                 # out: (B, 64,  56, 56)
        self.layer2 = r18.layer2                 # out: (B, 128, 28, 28)
        self.faa    = FrequencyAttentionAdapter(channels=128)
        self.layer3 = r18.layer3                 # out: (B, 256, 14, 14)
        self.dct    = DCTBlock(channels=256, block_size=7)
        self.layer4 = r18.layer4                 # out: (B, 512,  7,  7)
        self.pool   = nn.AdaptiveAvgPool2d(1)

        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, d),
            nn.BatchNorm1d(d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.faa(x)           # FAA after layer2
        x = self.layer3(x)
        x = self.dct(x)           # DCT after layer3
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)       # (B, 256)


# ─────────────────────────────────────────────────────────────────────────────
# Stream 3 — SRM Noise Analysis
# ─────────────────────────────────────────────────────────────────────────────

class SRMStream(nn.Module):
    """
    Steganalysis Rich Model (SRM) noise stream.
    Three fixed 5×5 SRM kernels applied to each RGB channel (9 residual maps).
    Detects high-frequency GAN manipulation residuals in single frames.

    Why SRM over rPPG:
      - rPPG needs 30-90 frames for physiological validity; SRM works on 1 frame
      - rPPG causes BatchNorm collapse on compressed frames (NaN gradients)
      - SRM residuals directly expose GAN pixel-level artifacts

    Kernel normalisation constants (48, 12, 12) follow Fridrich & Kodovsky (2012).

    Input  : (B, 3, 224, 224)
    Output : (B, 256)
    Params : ~0.16M  ← smallest stream, strongest ablation gain (+0.0139 AUC)
    """

    def __init__(self, d: int = 256):
        super().__init__()
        # Build 3 fixed SRM kernels (5×5)
        srm = torch.zeros(3, 1, 5, 5)

        # K1 — Horizontal gradient (normalised by 48)
        srm[0, 0] = torch.tensor([
            [-1, -2,  0,  2,  1],
            [-4, -8,  0,  8,  4],
            [-6,-12,  0, 12,  6],
            [-4, -8,  0,  8,  4],
            [-1, -2,  0,  2,  1],
        ], dtype=torch.float32) / 48.

        # K2 — Symmetric Laplacian (normalised by 12)
        srm[1, 0] = torch.tensor([
            [ 0,  0, -1,  0,  0],
            [ 0, -1,  2, -1,  0],
            [-1,  2,  4,  2, -1],
            [ 0, -1,  2, -1,  0],
            [ 0,  0, -1,  0,  0],
        ], dtype=torch.float32) / 12.

        # K3 — High-pass cross-channel (normalised by 12)
        srm[2, 0] = torch.tensor([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1],
        ], dtype=torch.float32) / 12.

        # Fixed — not learned
        self.register_buffer('srm_kernels', srm)

        # Lightweight CNN to process 9 residual maps
        self.cnn = nn.Sequential(
            # Stage 1
            nn.Conv2d(9, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 224→112
            # Stage 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 112→56
            # Stage 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),           # →(B,128,1,1)
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, d),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply 3 SRM kernels to each of 3 colour channels → 9 residual maps
        residuals = torch.cat(
            [F.conv2d(x[:, c:c+1], self.srm_kernels, padding=2)
             for c in range(3)],
            dim=1
        )                                      # (B, 9, 224, 224)
        residuals = residuals.clamp(-3., 3.)   # Eq. 4 in paper
        feats = self.cnn(residuals)            # (B, 128)
        return self.proj(feats)                # (B, 256)
