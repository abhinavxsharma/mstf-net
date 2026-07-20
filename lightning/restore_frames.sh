#!/bin/bash
# ============================================================
# lightning/restore_frames.sh
# Restore WildDeepfake frames from Drive zip to local SSD
#
# Drive file : gdrive:wilddeepfake_frames.zip
# Every session: ~5-8 min restore (much smaller than 72GB raw)
# ============================================================

set -e

LOCAL_SSD="/teamspace/studios/this_studio"
FRAMES_DIR="$LOCAL_SSD/wilddeepfake"
ZIP_LOCAL="$LOCAL_SSD/wilddeepfake_frames.zip"
DRIVE_ZIP="gdrive:wilddeepfake_frames.zip"

echo "=================================================="
echo "Restoring WildDeepfake frames from Drive"
echo "=================================================="

# ── Already on SSD? ───────────────────────────────────────────
# Check ALL FOUR split folders — a partial/interrupted prior
# extraction can leave train/real populated while train/fake,
# test/real, test/fake are still empty. Checking only train/real
# would falsely report "already restored" and skip the real fix.
R_TR=$(find "$FRAMES_DIR/train/real" -name "*.png" 2>/dev/null | wc -l || echo 0)
F_TR=$(find "$FRAMES_DIR/train/fake" -name "*.png" 2>/dev/null | wc -l || echo 0)
R_TE=$(find "$FRAMES_DIR/test/real"  -name "*.png" 2>/dev/null | wc -l || echo 0)
F_TE=$(find "$FRAMES_DIR/test/fake"  -name "*.png" 2>/dev/null | wc -l || echo 0)
if [ "$R_TR" -gt 1000 ] && [ "$F_TR" -gt 1000 ] && [ "$R_TE" -gt 500 ] && [ "$F_TE" -gt 500 ]; then
    echo "✅ Frames already on SSD (train: real=$R_TR fake=$F_TR | test: real=$R_TE fake=$F_TE)"
    exit 0
fi
echo "Frames incomplete on SSD (train: real=$R_TR fake=$F_TR | test: real=$R_TE fake=$F_TE) — restoring..."
# Clear any partial/stale extraction so we don't merge with a fresh unzip
rm -rf "$FRAMES_DIR"

# ── rclone configured? ────────────────────────────────────────
if ! rclone listremotes 2>/dev/null | grep -q "gdrive:"; then
    echo "❌ rclone not configured — run bash lightning/setup_rclone.sh"
    exit 1
fi

# ── Check Drive zip exists ────────────────────────────────────
if ! rclone ls "$DRIVE_ZIP" 2>/dev/null | grep -q "wilddeepfake"; then
    echo "❌ wilddeepfake_frames.zip not on Drive"
    echo "   Run first: python scripts/download_wilddeepfake.py"
    exit 1
fi

SIZE=$(rclone ls "$DRIVE_ZIP" | awk '{printf "%.1fGB", $1/1073741824}')
echo "Drive zip: $SIZE"

# ── Download zip ──────────────────────────────────────────────
echo ""
echo "[1/3] Downloading zip from Drive..."
t1=$(date +%s)
rclone copy "$DRIVE_ZIP" "$LOCAL_SSD/" \
    --progress --transfers 4 --drive-chunk-size 256M --stats 20s
t2=$(date +%s)
echo "✅ Downloaded in $(( (t2-t1)/60 ))m $(( (t2-t1)%60 ))s"

# ── Unzip ─────────────────────────────────────────────────────
echo ""
echo "[2/3] Unzipping..."
mkdir -p "$FRAMES_DIR"
t1=$(date +%s)
unzip -q "$ZIP_LOCAL" -d "$FRAMES_DIR/"
t2=$(date +%s)
rm -f "$ZIP_LOCAL"
echo "✅ Extracted in $(( (t2-t1)/60 ))m $(( (t2-t1)%60 ))s"

# ── Verify ────────────────────────────────────────────────────
echo ""
echo "[3/3] Verifying..."
R_TR=$(find "$FRAMES_DIR/train/real" -name "*.png" 2>/dev/null | wc -l || echo 0)
F_TR=$(find "$FRAMES_DIR/train/fake" -name "*.png" 2>/dev/null | wc -l || echo 0)
R_TE=$(find "$FRAMES_DIR/test/real"  -name "*.png" 2>/dev/null | wc -l || echo 0)
F_TE=$(find "$FRAMES_DIR/test/fake"  -name "*.png" 2>/dev/null | wc -l || echo 0)

echo "  Train: real=$R_TR  fake=$F_TR"
echo "  Test : real=$R_TE  fake=$F_TE"

if [ "$R_TR" -lt 1000 ]; then
    echo "❌ Too few frames — zip may be incomplete"
    exit 1
fi

echo ""
echo "✅ Frames ready at $FRAMES_DIR"
echo "=================================================="