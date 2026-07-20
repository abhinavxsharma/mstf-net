#!/bin/bash
# ============================================================
# lightning/train_session.sh
# MSTF-Net — WildDeepfake Training on Lightning AI
#
# Session 1: Restore frames + train seeds 42, 123, 456
# Session 2: Train seeds 789, 1011 + eval + upload
#
# Usage:
#   export WANDB_API_KEY=your_key
#   export HF_TOKEN=your_token
#   bash lightning/train_session.sh --session 1
#   bash lightning/train_session.sh --session 2
# ============================================================

set -e

SESSION=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --session) SESSION="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "=================================================="
echo "MSTF-Net — WildDeepfake Training"
echo "Session : $SESSION"
echo "Started : $(date)"
echo "=================================================="

export PYTHONPATH="$(pwd):$PYTHONPATH"

LOCAL_SSD="/teamspace/studios/this_studio"
FRAMES_DIR="$LOCAL_SSD/wilddeepfake"
CKPT_DIR="$LOCAL_SSD/MSTF_checkpoints_wild"
RESULTS_DIR="$LOCAL_SSD/MSTF_results_wild"
DRIVE_CKPTS="gdrive:MSTF_checkpoints_wild"
DRIVE_RESULTS="gdrive:MSTF_results_wild"

mkdir -p "$CKPT_DIR" "$RESULTS_DIR"

# ── Helpers ───────────────────────────────────────────────────
sync_ckpts() {
    echo "  → Syncing checkpoints to Drive..."
    rclone sync "$CKPT_DIR" "$DRIVE_CKPTS" \
        --transfers 4 --stats 10s 2>/dev/null
    echo "  ✅ Checkpoints safe on Drive"
}

restore_ckpts() {
    echo "  → Restoring checkpoints from Drive..."
    mkdir -p "$CKPT_DIR"
    rclone copy "$DRIVE_CKPTS" "$CKPT_DIR" --transfers 4 2>/dev/null
    COUNT=$(ls "$CKPT_DIR"/*.pth 2>/dev/null | wc -l || echo 0)
    echo "  ✅ $COUNT checkpoints restored"
}

train_seed() {
    SEED=$1
    NAME="mstfnet_wilddeepfake_seed${SEED}_BEST.pth"
    if [ -f "$CKPT_DIR/$NAME" ]; then
        echo "  ✅ seed$SEED already done — skipping"
        return 0
    fi
    echo "  Training seed $SEED..."
    python scripts/train.py \
        --config configs/wilddeepfake.yaml \
        --seed "$SEED"
    sync_ckpts
}

# ══════════════════════════════════════════════════════════════
# SESSION 1 — Restore frames + Train seeds 42, 123, 456
# Timeline:
#   ~8  min  restore frames from Drive zip
#   ~50 min  seed 42
#   ~50 min  seed 123
#   ~50 min  seed 456
#   Total: ~2.8 hrs ✅
# ══════════════════════════════════════════════════════════════
if [ "$SESSION" == "1" ]; then

    echo ""
    echo "─── Phase 1/4: Restore Frames ─────────────────────"
    bash lightning/restore_frames.sh

    echo ""
    echo "─── Phase 2/4: Train seed 42 ───────────────────────"
    train_seed 42

    echo ""
    echo "─── Phase 3/4: Train seed 123 ──────────────────────"
    train_seed 123

    echo ""
    echo "─── Phase 4/4: Train seed 456 ──────────────────────"
    train_seed 456

    sync_ckpts

    echo ""
    echo "=================================================="
    echo "SESSION 1 COMPLETE ✅"
    echo ""
    echo "Live results: https://wandb.ai/i-abhinavxsharma/mstf-net"
    echo ""
    echo "Start Session 2 in new Lightning studio:"
    echo "  export WANDB_API_KEY=your_key"
    echo "  export HF_TOKEN=your_token"
    echo "  bash lightning/setup.sh"
    echo "  bash lightning/train_session.sh --session 2"
    echo "=================================================="
fi

# ══════════════════════════════════════════════════════════════
# SESSION 2 — Train seeds 789, 1011 + Evaluate + Upload
# Timeline:
#   ~8  min  restore frames
#   ~5  min  restore checkpoints
#   ~50 min  seed 789
#   ~50 min  seed 1011
#   ~15 min  evaluate + upload
#   Total: ~2.2 hrs ✅
# ══════════════════════════════════════════════════════════════
if [ "$SESSION" == "2" ]; then

    echo ""
    echo "─── Phase 1/5: Restore Frames ─────────────────────"
    bash lightning/restore_frames.sh

    echo ""
    echo "─── Phase 2/5: Restore Checkpoints ────────────────"
    restore_ckpts

    # Verify Session 1
    MISSING=0
    for SEED in 42 123 456; do
        F="$CKPT_DIR/mstfnet_wilddeepfake_seed${SEED}_BEST.pth"
        [ -f "$F" ] && echo "  ✅ seed$SEED" || { echo "  ❌ seed$SEED MISSING"; MISSING=$((MISSING+1)); }
    done
    [ "$MISSING" -gt 0 ] && { echo "❌ $MISSING checkpoints missing from Session 1"; exit 1; }

    echo ""
    echo "─── Phase 3/5: Train seed 789 ──────────────────────"
    train_seed 789

    echo ""
    echo "─── Phase 4/5: Train seed 1011 ─────────────────────"
    train_seed 1011

    echo ""
    echo "─── Phase 5/5: Evaluate + Upload ───────────────────"
    python scripts/evaluate.py \
        --checkpoint_dir "$CKPT_DIR" \
        --frames_root    "$FRAMES_DIR" \
        --results_dir    "$RESULTS_DIR" \
        --dataset        wilddeepfake \
        --aggregate

    # Print final AUC
    python3 -c "
import json, os
f = '$RESULTS_DIR/final_results.json'
if os.path.exists(f):
    r = json.load(open(f))
    print()
    print('  ╔══════════════════════════════════════╗')
    print(f'  ║  FINAL AUC : {r[\"formatted\"]}        ║')
    print(f'  ║  Seeds     : {r[\"n_seeds\"]}                   ║')
    print(f'  ║  Acc       : {r[\"mean_acc\"]:.4f}              ║')
    print(f'  ║  F1        : {r[\"mean_f1\"]:.4f}              ║')
    print('  ╚══════════════════════════════════════╝')
" 2>/dev/null

    # Upload to HuggingFace
    [ -n "$HF_TOKEN" ] && python scripts/upload_weights.py \
        --checkpoint_dir "$CKPT_DIR" \
        --results_dir    "$RESULTS_DIR" \
        --repo_id        "ixabhinavsharma/mstf-net"

    # Sync to Drive
    sync_ckpts
    rclone sync "$RESULTS_DIR" "$DRIVE_RESULTS" --transfers 4 2>/dev/null

    echo ""
    echo "=================================================="
    echo "ALL TRAINING COMPLETE ✅"
    echo ""
    echo "W&B  : https://wandb.ai/i-abhinavxsharma/mstf-net"
    echo "HF   : https://huggingface.co/ixabhinavsharma/mstf-net"
    echo ""
    echo "Now push to GitHub:"
    echo "  git add . && git commit -m 'WildDeepfake results'"
    echo "  git push"
    echo "=================================================="
fi

echo "Finished: $(date)"
