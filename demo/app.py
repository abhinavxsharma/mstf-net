"""
demo/app.py
===========
MSTF-Net Gradio Demo — hosted on HuggingFace Spaces.
Upload any video → get REAL/FAKE prediction with confidence,
per-frame breakdown, and DSTG gating weight visualisation.

Deploy: Push this file to HuggingFace Space (SDK: Gradio)
        Weights auto-download from ixabhinavsharma/mstf-net on first run.

Authors : Abhinav Vats et al., Chandigarh University
Accepted: ISSMAD 2026 (Google + IEEE Signal Processing Society)
"""

import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch

# ── Auto-download weights from HuggingFace Hub ───────────────
REPO_ID   = 'ixabhinavsharma/mstf-net'
CKPT_FILE = 'weights/mstfnet_wilddeepfake_seed789_BEST.pth'
LOCAL_DIR = Path('weights')
LOCAL_DIR.mkdir(exist_ok=True)
LOCAL_CKPT = LOCAL_DIR / 'mstfnet_best.pth'

def download_weights():
    if LOCAL_CKPT.exists():
        return
    print('Downloading MSTF-Net weights from HuggingFace...')
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=CKPT_FILE,
            local_dir=str(LOCAL_DIR),
        )
        import shutil
        shutil.copy(path, str(LOCAL_CKPT))
        print('✅ Weights downloaded')
    except Exception as e:
        print(f'⚠️  Weight download failed: {e}')
        print('Running in demo mode without weights')

# ── Load model ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
download_weights()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL  = None

def get_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    try:
        from mstfnet import MSTFNet, load_checkpoint
        m = MSTFNet().to(DEVICE)
        if LOCAL_CKPT.exists():
            load_checkpoint(m, str(LOCAL_CKPT), DEVICE)
        m.eval()
        MODEL = m
        print('✅ Model loaded')
    except Exception as e:
        print(f'Model load error: {e}')
    return MODEL


# ── Inference function ────────────────────────────────────────
def predict(video_path):
    if video_path is None:
        return 'Please upload a video', None, None

    model = get_model()
    if model is None:
        return 'Model not loaded', None, None

    try:
        from mstfnet import VideoDataset
        ds     = VideoDataset([str(video_path)], n_frames=8)
        frames, _ = ds[0]
        frames = frames.to(DEVICE)

        per_frame_probs  = []
        per_frame_alpha  = []
        per_frame_qn     = []

        with torch.no_grad():
            for i in range(frames.size(0)):
                frame = frames[i].unsqueeze(0)
                logits, alpha, q_n = model(frame)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                per_frame_probs.append(prob)
                per_frame_alpha.append(alpha[0].cpu().numpy())
                per_frame_qn.append(q_n[0].item())

        mean_prob  = float(np.mean(per_frame_probs))
        label      = '🔴 FAKE' if mean_prob >= 0.5 else '🟢 REAL'
        mean_alpha = np.mean(per_frame_alpha, axis=0)
        mean_qn    = float(np.mean(per_frame_qn))

        # Build result text
        result_text = (
            f'**Prediction: {label}**\n\n'
            f'Fake probability: {mean_prob*100:.1f}%\n'
            f'Quality score Qn: {mean_qn:.3f} '
            f'(0.78=high quality, 0.33=compressed)\n\n'
            f'**DSTG Stream Weights:**\n'
            f'- Spatial  (EfficientNet-B0): {mean_alpha[0]:.3f}\n'
            f'- Spectral (ResNet-18+DCT):   {mean_alpha[1]:.3f}\n'
            f'- SRM Noise:                  {mean_alpha[2]:.3f}\n\n'
            f'**Per-frame fake probabilities:**\n'
            f'{[f"{p:.3f}" for p in per_frame_probs]}'
        )

        # Build per-frame bar chart
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Per-frame confidence
        frames_idx = list(range(1, len(per_frame_probs) + 1))
        colors = ['#e74c3c' if p >= 0.5 else '#2ecc71' for p in per_frame_probs]
        ax1.bar(frames_idx, per_frame_probs, color=colors, alpha=0.8)
        ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='threshold')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Fake Probability')
        ax1.set_title('Per-Frame Confidence')
        ax1.set_ylim(0, 1)
        ax1.legend()

        # DSTG weights pie chart
        stream_names  = ['Spatial\n(EfficientNet)', 'Spectral\n(ResNet+DCT)', 'SRM\nNoise']
        stream_colors = ['#3498db', '#9b59b6', '#1abc9c']
        ax2.pie(
            mean_alpha,
            labels=stream_names,
            colors=stream_colors,
            autopct='%1.1f%%',
            startangle=90,
        )
        ax2.set_title(f'DSTG Stream Weights\n(Qn={mean_qn:.2f})')

        plt.tight_layout()

        # Save plot
        plot_path = tempfile.mktemp(suffix='.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return result_text, plot_path, mean_prob

    except Exception as e:
        return f'Error: {e}', None, None


# ── Gradio UI ────────────────────────────────────────────────
with gr.Blocks(
    title='MSTF-Net Deepfake Detector',
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown("""
    # 🔍 MSTF-Net Deepfake Detector
    **Adaptive Multi-Stream Deepfake Detection via Dynamic Spectral-Temporal Gating**

    Accepted at [ISSMAD 2026](https://rsvp.withgoogle.com/events/issmad2026)
    — co-sponsored by **Google** + **IEEE Signal Processing Society**

    📄 [Paper](https://github.com/abhinavxsharma/mstf-net) |
    💻 [GitHub](https://github.com/abhinavxsharma/mstf-net) |
    🤗 [Weights](https://huggingface.co/ixabhinavsharma/mstf-net)

    ---
    **AUC 0.9792 ± 0.0025** on DeeperForensics-1.0 |
    **3.0× better** compression robustness than baseline
    """)

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(
                label='Upload Video',
                height=300,
            )
            submit_btn = gr.Button('🔍 Detect Deepfake', variant='primary')

            gr.Examples(
                examples=[],
                inputs=video_input,
                label='Example Videos',
            )

        with gr.Column(scale=1):
            result_text = gr.Markdown(label='Result')
            confidence  = gr.Slider(
                minimum=0, maximum=1, value=0,
                label='Fake Probability', interactive=False
            )

    plot_output = gr.Image(label='Frame Analysis & DSTG Weights', height=350)

    submit_btn.click(
        fn=predict,
        inputs=[video_input],
        outputs=[result_text, plot_output, confidence],
    )

    gr.Markdown("""
    ---
    ### How It Works
    MSTF-Net processes **8 uniformly sampled frames** through three parallel streams:
    1. **Spatial Stream** (EfficientNet-B0) — detects blending seams & texture artifacts
    2. **Spectral Stream** (ResNet-18 + FAA + DCT) — detects GAN frequency fingerprints
    3. **SRM Noise Stream** (fixed filters) — detects high-frequency manipulation residuals

    The **DSTG module** adaptively weights these streams based on video compression quality
    (Laplacian quality score Qn), suppressing unreliable streams under heavy compression.

    *Authors: Abhinav Vats, Poonam Jyoti, Ishika Bhardwaj, Tanvi Garg, Tannu Ghanghas*
    *Chandigarh University, Mohali, Punjab, India*
    """)


if __name__ == '__main__':
    demo.launch(share=True)