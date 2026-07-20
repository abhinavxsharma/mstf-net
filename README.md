# MSTF-Net 🔍

**Adaptive Multi-Stream Deepfake Detection via Dynamic Spectral-Temporal Gating**

[![CI](https://github.com/abhinavxsharma/mstf-net/actions/workflows/ci.yml/badge.svg)](https://github.com/abhinavxsharma/mstf-net/actions)
[![HuggingFace](https://img.shields.io/badge/🤗%20Weights-ixabhinavsharma/mstf--net-yellow)](https://huggingface.co/ixabhinavsharma/mstf-net)
[![Demo](https://img.shields.io/badge/🤗%20Demo-Spaces-blue)](https://huggingface.co/spaces/ixabhinavsharma/mstf-net-demo)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-orange)](https://pytorch.org)
[![License MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ISSMAD 2026](https://img.shields.io/badge/ISSMAD-2026-red)](https://rsvp.withgoogle.com/events/issmad2026)

> Accepted at **ISSMAD 2026** — International Symposium on Synthetic Media Attribution and Detection,
> co-sponsored by **Google** and the **IEEE Signal Processing Society**, San Francisco Bay Area, August 2026.

---

## Results

| Dataset | AUC | Params |
|---|---|---|
| **DeeperForensics-1.0** | **0.9792 ± 0.0025** | 29.6M |
| FF++ C23 | 0.7269 ± 0.0101 | 29.6M |
| FF++ C40 | 0.7143 ± 0.0103 | 29.6M |

**3.0× better compression robustness** than EfficientNet-B0 baseline (C23→C40 drop: −0.0126 vs −0.0372).
Competitive with CLIP-large (0.9800) at **13× fewer parameters** (29.6M vs ~400M).

---

## Architecture

MSTF-Net processes each frame through **three parallel streams** fused by a quality-aware gating module:

```
Input Frame (224×224)
       │
       ├──► Stream 1: SpatialStream  (EfficientNet-B0)    → f_s (256-d)  ─┐
       ├──► Stream 2: SpectralStream (ResNet-18+FAA+DCT)  → f_f (256-d)  ─┼──► DSTG ──► Classifier
       ├──► Stream 3: SRMStream      (fixed SRM filters)  → f_n (256-d)  ─┘
       └──► LaplacianQE              → Q_n ∈ (0,1)        ────────────────┘
```

**Key novelty:** The DSTG module conditions per-stream fusion weights on a differentiable
Laplacian quality score Q_n. Under heavy compression (C40, Q_n≈0.33), DSTG suppresses
the unreliable spectral and SRM streams, shifting weight to the robust spatial stream.
Under high quality (C23, Q_n≈0.78), all three streams contribute equally.

---

## Ablation Study (FF++ C23, Identity-Safe Split)

| Configuration | AUC | F1 | Params |
|---|---|---|---|
| Spatial only | 0.7409 ± 0.0142 | 0.635 | 4.8M |
| SRM only | 0.5840 ± 0.0474 | 0.551 | 0.2M |
| Spectral only | 0.6162 ± 0.0345 | 0.510 | 14.6M |
| Spatial + Spectral | 0.7365 ± 0.0062 | 0.605 | 19.4M |
| **Spatial + SRM** | **0.7548 ± 0.0076** | **0.651** | **5.0M** |
| All streams, static | 0.7369 ± 0.0058 | 0.632 | 19.5M |
| DSTG, Qn≡1 | 0.7332 ± 0.0099 | 0.624 | 19.8M |
| **MSTF-Net v3 (ours)** | **0.7373 ± 0.0101** | **0.636** | **29.6M** |

SRM is the strongest single contributor (+0.0139 AUC over spatial alone).
MSTF-Net achieves the highest mean AUC among all three-stream configurations.

---

## Quick Install

```bash
git clone https://github.com/abhinavxsharma/mstf-net
cd mstf-net
pip install -e .
pip install -r requirements.txt
```

---

## Quick Start

```python
from mstfnet import MSTFNet
import torch

# Load model
model = MSTFNet()

# Run inference on a batch of frames
x = torch.randn(4, 3, 224, 224)   # 4 frames
logits, alpha, q_n = model(x)

# Get fake probability
probs = model.predict_proba(x)
print(f'Fake probabilities: {probs}')
print(f'DSTG weights (spatial/spectral/srm): {alpha}')
print(f'Quality scores: {q_n}')
```

---

## Run Inference on Any Video

```bash
python scripts/inference.py \
    --video path/to/video.mp4 \
    --checkpoint weights/mstfnet_deeperforensics_seed42_BEST.pth
```

Output:
```
Prediction : FAKE  (87.3% fake)
Quality Qn : 0.412  (0.78=C23 quality, 0.33=C40 quality)
DSTG weights:
  Spatial  : 0.541
  Spectral : 0.231
  SRM      : 0.228
Per-frame  : [0.821, 0.934, 0.756, 0.891, 0.923, 0.812, 0.901, 0.845]
```

---

## Train on DeeperForensics-1.0

**Session 1** (extract frames + train seeds 42 & 123):
```bash
# On Lightning AI Studio
bash lightning/setup.sh
export WANDB_API_KEY=your_key
export HF_TOKEN=your_token
bash lightning/train_session.sh --session 1
```

**Session 2** (train seed 456 + evaluate + upload):
```bash
bash lightning/setup.sh
bash lightning/train_session.sh --session 2
```

All checkpoints auto-save to Google Drive. Sessions can be interrupted safely.

---

## Run Tests

```bash
pytest tests/ -v
```

All 13 unit tests cover stream output shapes, SRM kernel fixedness,
quality estimator bounds, DSTG simplex constraint, and all 8 ablation configs.

---

## Citation

If you use MSTF-Net in your research, please cite:

```bibtex
@inproceedings{vats2026mstfnet,
  title     = {MSTF-Net: Adaptive Multi-Stream Deepfake Detection via
               Dynamic Spectral-Temporal Gating},
  author    = {Vats, Abhinav and Jyoti, Poonam and Bhardwaj, Ishika and
               Garg, Tanvi and Ghanghas, Tannu},
  booktitle = {International Symposium on Synthetic Media Attribution
               and Detection (ISSMAD)},
  year      = {2026},
  note      = {Co-sponsored by Google and IEEE Signal Processing Society}
}
```

---

## Authors

**Abhinav Vats**, Poonam Jyoti, Ishika Bhardwaj, Tanvi Garg, Tannu Ghanghas

Department of Computer Science and Engineering (AI & ML),
Chandigarh University, Mohali, Punjab 140413, India

📧 vats.abhinav247@gmail.com |
🐙 [GitHub](https://github.com/abhinavxsharma) |
🤗 [HuggingFace](https://huggingface.co/ixabhinavsharma) |
📊 [W&B](https://wandb.ai/i-abhinavxsharma)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
