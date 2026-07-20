#!/bin/bash
# ============================================================
# lightning/save_rclone_config.sh
# ONE-TIME: Save rclone config to Drive so every future session
# can restore it in 10 seconds without re-doing OAuth
#
# Run this ONCE after rclone is configured and working
# Then update RCLONE_CONF_FILE_ID in setup.sh
# ============================================================

set -e

echo "=================================================="
echo "Saving rclone config to Drive"
echo "=================================================="

# Verify rclone is working
if ! rclone listremotes 2>/dev/null | grep -q "gdrive:"; then
    echo "❌ rclone gdrive not configured yet"
    echo "   Run: bash lightning/setup_rclone.sh"
    exit 1
fi
echo "✅ rclone gdrive is working"

# Check config file exists
if [ ! -f ~/.config/rclone/rclone.conf ]; then
    echo "❌ rclone.conf not found at ~/.config/rclone/rclone.conf"
    exit 1
fi

echo ""
echo "Saving config to Drive..."
rclone copy ~/.config/rclone/rclone.conf "gdrive:rclone_backup/rclone.conf"
echo "✅ Saved to gdrive:rclone_backup/rclone.conf"

# Get the file ID
echo ""
echo "Getting file ID..."
rclone lsf "gdrive:rclone_backup/" --format "ip" 2>/dev/null | grep "rclone.conf" && \
    echo "  (copy the ID above)" || \
    echo "  Open drive.google.com → rclone_backup → rclone.conf → right click → Get link → copy ID"

echo ""
echo "=================================================="
echo "IMPORTANT — Do this now:"
echo ""
echo "1. Open drive.google.com in browser"
echo "2. Find folder: rclone_backup"
echo "3. Right-click rclone.conf → Get link"
echo "4. Copy the FILE ID from the URL:"
echo "   URL looks like: https://drive.google.com/file/d/XXXXXXXX/view"
echo "   The XXXXXXXX part is your file ID"
echo ""
echo "5. Open lightning/setup.sh"
echo "6. Replace YOUR_RCLONE_CONF_ID_HERE with your file ID"
echo ""
echo "After that, every session just needs:"
echo "   bash lightning/setup.sh"
echo "   (rclone restores in 10 seconds automatically)"
echo "=================================================="
