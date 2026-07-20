#!/bin/bash
# ============================================================
# lightning/setup.sh
# Run FIRST every Lightning AI session (~5 min)
#
# rclone strategy:
#   Config saved to Drive once → restored every session via gdown
#   Takes 10 seconds instead of full OAuth flow
#
# SECURITY: Never hardcode API keys. Set in terminal:
#   export WANDB_API_KEY=your_key
#   export HF_TOKEN=your_token
#
# RCLONE CONFIG FILE ID (from your Drive):
#   After first time saving, set RCLONE_CONF_FILE_ID below
# ============================================================

set -e

# ── SET THIS AFTER FIRST SAVE ────────────────────────────────
# Run once: rclone copy ~/.config/rclone/rclone.conf "gdrive:rclone_backup/rclone.conf"
# Then open drive.google.com → rclone_backup/rclone.conf → right click → Get link
# Copy the ID from the URL (between /d/ and /view) and paste below
RCLONE_CONF_FILE_ID="YOUR_RCLONE_CONF_ID_HERE"
# Example: RCLONE_CONF_FILE_ID="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs"

echo "=================================================="
echo "MSTF-Net — Lightning AI Setup"
echo "Started: $(date)"
echo "=================================================="

# ── Install packages ──────────────────────────────────────────
echo ""
echo "[1/5] Installing packages..."
pip install -q --upgrade pip
pip install -q \
    torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -q \
    timm==0.9.16 \
    opencv-python-headless==4.9.0.80 \
    scikit-learn==1.4.2 \
    wandb==0.17.0 \
    huggingface_hub==0.23.4 \
    pyyaml==6.0.1 \
    tqdm==4.66.4 \
    gradio==4.36.1 \
    matplotlib==3.9.0 \
    pillow==10.3.0 \
    gdown
echo "✅ Packages installed"

# ── Install rclone ────────────────────────────────────────────
echo ""
echo "[2/5] Setting up rclone..."
if ! command -v rclone &> /dev/null; then
    curl https://rclone.org/install.sh | sudo bash 2>/dev/null
fi
echo "  rclone: $(rclone --version | head -1)"

# Fix config dir ownership (known Lightning bug)
if [ -d ~/.config/rclone ]; then
    OWNER=$(stat -c '%U' ~/.config/rclone 2>/dev/null || echo "unknown")
    if [ "$OWNER" != "$USER" ] && [ "$OWNER" != "unknown" ]; then
        rmdir ~/.config/rclone 2>/dev/null || true
    fi
fi
mkdir -p ~/.config/rclone

# Restore rclone config from Drive
if rclone listremotes 2>/dev/null | grep -q "gdrive:"; then
    echo "  ✅ rclone gdrive already working"
elif [ "$RCLONE_CONF_FILE_ID" != "YOUR_RCLONE_CONF_ID_HERE" ]; then
    echo "  Restoring rclone config from Drive..."
    gdown "$RCLONE_CONF_FILE_ID" -O ~/.config/rclone/rclone.conf -q
    if rclone listremotes 2>/dev/null | grep -q "gdrive:"; then
        echo "  ✅ rclone gdrive restored in seconds"
    else
        echo "  ❌ Config restore failed — run setup_rclone.sh"
    fi
else
    echo "  ⚠️  RCLONE_CONF_FILE_ID not set in setup.sh"
    echo "  First time: run bash lightning/save_rclone_config.sh"
fi

# ── Verify GPU ────────────────────────────────────────────────
echo ""
echo "[3/5] GPU check..."
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory/1e9
    print(f'  ✅ {name} — {vram:.0f}GB VRAM')
else:
    print('  ❌ No GPU found')
    exit(1)
"

# ── Check SSD ─────────────────────────────────────────────────
echo ""
echo "[4/5] Storage check..."
AVAIL=$(df -h /teamspace/studios/this_studio | awk 'NR==2{print $4}')
echo "  ✅ /teamspace/studios/this_studio — ${AVAIL} available"

# ── API logins ────────────────────────────────────────────────
echo ""
echo "[5/5] API logins..."
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY" --relogin 2>/dev/null && echo "  ✅ W&B logged in"
else
    echo "  ⚠️  WANDB_API_KEY not set"
fi
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null && echo "  ✅ HuggingFace logged in"
else
    echo "  ⚠️  HF_TOKEN not set"
fi

echo ""
echo "=================================================="
echo "✅ Setup complete"
echo ""
echo "Next:"
echo "  bash lightning/train_session.sh --session 1"
echo "  bash lightning/train_session.sh --session 2"
echo "=================================================="
