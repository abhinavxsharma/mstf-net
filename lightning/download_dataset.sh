#!/bin/bash
# ============================================================
# lightning/download_dataset.sh
# ONE-TIME dataset setup for Lightning AI
#
# What this does (only needs to run ONCE ever):
#   1. Download DeeperForensics zips from Drive → local SSD
#   2. Extract 8 frames per video on local SSD
#   3. Upload ONLY the frames (not zips) back to Drive
#      Frames = ~8GB vs zips = 230GB — much faster to restore
#
# Every future session:
#   train_session.sh automatically restores frames from Drive
#   Takes ~35 min (8GB) not ~90 min (230GB)
#
# NEVER needs to run again unless you delete gdrive:DF_Frames
# ============================================================

set -e

LOCAL_SSD="/teamspace/studios/this_studio"
LOCAL_DATASET="$LOCAL_SSD/DeeperForensics-1.0"
LOCAL_FRAMES="$LOCAL_SSD/DF_Frames"
LOCAL_TMP="$LOCAL_SSD/df_tmp"
DRIVE_DATASET="gdrive:DeeperForensics-1.0"
DRIVE_FRAMES="gdrive:DF_Frames"

echo "=================================================="
echo "DeeperForensics-1.0 — One-Time Frame Setup"
echo "=================================================="
echo ""
echo "Storage plan:"
echo "  Download : Drive zips (230GB) → local SSD    (~90 min)"
echo "  Extract  : 8 frames/video on local SSD        (~55 min)"
echo "  Upload   : Frames only (~8GB) → Drive         (~35 min)"
echo "  Future   : Restore frames from Drive each session (~35 min)"
echo ""
echo "Total today : ~3 hrs"
echo "Future cost : ~35 min per session (frames only)"
echo "=================================================="

# ── Verify rclone ─────────────────────────────────────────────
if ! rclone listremotes 2>/dev/null | grep -q "gdrive:"; then
    echo "❌ rclone gdrive not configured"
    echo "   Run: bash lightning/setup_rclone.sh"
    exit 1
fi
echo "✅ rclone gdrive ready"

# ── Check if frames already on Drive (already done before) ────
echo ""
echo "[CHECK] Frames on Drive?"
DRIVE_COUNT=$(rclone ls "$DRIVE_FRAMES/train/real" 2>/dev/null | wc -l || echo 0)

if [ "$DRIVE_COUNT" -gt 50000 ]; then
    echo "✅ Frames already on Drive ($DRIVE_COUNT files in train/real)"
    echo ""
    echo "download_dataset.sh already completed before."
    echo "No need to run again."
    echo ""
    echo "To start training:"
    echo "  bash lightning/train_session.sh --session 1"
    exit 0
fi

echo "  Frames not on Drive yet — running full setup"

# ── Check disk space ──────────────────────────────────────────
echo ""
AVAIL=$(df -h "$LOCAL_SSD" | awk 'NR==2{print $4}')
AVAIL_GB=$(df -BG "$LOCAL_SSD" | awk 'NR==2{print $4}' | tr -d 'G')
echo "[0] Disk space available: $AVAIL"
if [ "$AVAIL_GB" -lt 300 ]; then
    echo "⚠️  Need ~300GB free for zips + frames"
    echo "   Available: ${AVAIL_GB}GB"
    echo "   You have $(df -h "$LOCAL_SSD" | awk 'NR==2{print $2}') total"
fi

# ── Step 1: Download zips from Drive → local SSD ─────────────
echo ""
echo "[1/4] Downloading DeeperForensics zips from Drive..."
echo "  Source : $DRIVE_DATASET"
echo "  Dest   : $LOCAL_DATASET"
echo "  Size   : ~230GB"
echo "  Time   : ~90 min"
echo ""

mkdir -p "$LOCAL_DATASET"

# Check if already downloaded
ZIP_COUNT=$(ls "$LOCAL_DATASET"/*.zip 2>/dev/null | wc -l || echo 0)
if [ "$ZIP_COUNT" -ge 20 ]; then
    echo "  ✅ Zips already on local SSD ($ZIP_COUNT files) — skipping download"
else
    echo "  Starting download... (watch progress below)"
    rclone copy "$DRIVE_DATASET" "$LOCAL_DATASET" \
        --progress \
        --transfers 8 \
        --checkers 8 \
        --drive-chunk-size 128M \
        --stats 60s
    echo "  ✅ Download complete"
fi

echo ""
echo "  Files on local SSD:"
ls -lh "$LOCAL_DATASET/" | head -25

# ── Step 2: Extract frames ────────────────────────────────────
echo ""
echo "[2/4] Extracting frames..."
echo "  Protocol : 8 frames/video, 224×224, uniformly sampled"
echo "  Split    : DeeperForensics official train/val lists"
echo "  Time     : ~55 min"
echo ""

mkdir -p "$LOCAL_FRAMES" "$LOCAL_TMP"

# Check if already extracted
R_TR=$(find "$LOCAL_FRAMES/train/real" -name "*.jpg" 2>/dev/null | wc -l || echo 0)
F_TR=$(find "$LOCAL_FRAMES/train/fake" -name "*.jpg" 2>/dev/null | wc -l || echo 0)

if [ "$R_TR" -gt 50000 ] && [ "$F_TR" -gt 50000 ]; then
    echo "  ✅ Frames already extracted: real=$R_TR fake=$F_TR — skipping"
else
    python scripts/extract_frames.py \
        --dataset_dir "$LOCAL_DATASET" \
        --frames_out  "$LOCAL_FRAMES" \
        --local_tmp   "$LOCAL_TMP" \
        --n_frames    8 \
        --workers     16
fi

# Verify
R_TR=$(find "$LOCAL_FRAMES/train/real" -name "*.jpg" 2>/dev/null | wc -l || echo 0)
F_TR=$(find "$LOCAL_FRAMES/train/fake" -name "*.jpg" 2>/dev/null | wc -l || echo 0)
R_VA=$(find "$LOCAL_FRAMES/val/real"   -name "*.jpg" 2>/dev/null | wc -l || echo 0)
F_VA=$(find "$LOCAL_FRAMES/val/fake"   -name "*.jpg" 2>/dev/null | wc -l || echo 0)

echo ""
echo "  Frames extracted:"
echo "    Train: real=$R_TR  fake=$F_TR"
echo "    Val  : real=$R_VA  fake=$F_VA"

if [ "$R_TR" -lt 1000 ]; then
    echo "❌ Extraction failed — too few frames"
    exit 1
fi

# ── Step 3: Delete local zips to free SSD space ───────────────
echo ""
echo "[3/4] Freeing local SSD space (deleting zips)..."
echo "  Zips are still on Drive — not deleting from Drive"
rm -f "$LOCAL_DATASET"/*.zip
echo "  ✅ Local zips deleted (Drive copies safe)"
echo "  SSD now: $(df -h $LOCAL_SSD | awk 'NR==2{print $3}') used"

# ── Step 4: Upload frames to Drive ────────────────────────────
echo ""
echo "[4/4] Uploading frames to Drive for persistence..."
echo "  From : $LOCAL_FRAMES"
echo "  To   : $DRIVE_FRAMES"
echo "  Size : ~8GB (JPG frames only, not raw videos)"
echo "  Time : ~35 min"
echo ""
echo "  This is the LAST time you upload frames."
echo "  Future sessions restore from Drive in ~35 min."
echo ""

rclone sync "$LOCAL_FRAMES" "$DRIVE_FRAMES" \
    --progress \
    --transfers 16 \
    --checkers 16 \
    --stats 60s

echo ""
echo "✅ Frames on Drive: $DRIVE_FRAMES"

# Verify Drive upload
DRIVE_COUNT=$(rclone ls "$DRIVE_FRAMES/train/real" 2>/dev/null | wc -l || echo 0)
echo "  Drive has $DRIVE_COUNT files in train/real"

echo ""
echo "=================================================="
echo "ONE-TIME SETUP COMPLETE ✅"
echo ""
echo "Summary:"
echo "  Train: real=$R_TR  fake=$F_TR"
echo "  Val  : real=$R_VA  fake=$F_VA"
echo ""
echo "Frames permanently stored at: $DRIVE_FRAMES"
echo "  → Drive size used: ~8GB"
echo "  → Restore time next session: ~35 min (not 3.5 hrs)"
echo ""
echo "NEVER run this script again unless you delete gdrive:DF_Frames"
echo ""
echo "Next:"
echo "  bash lightning/train_session.sh --session 1"
echo "=================================================="
