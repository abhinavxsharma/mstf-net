#!/bin/bash
# ============================================================
# lightning/setup_rclone.sh
# ONE-TIME rclone Google Drive setup for Lightning AI
#
# Lightning has no browser — OAuth must be done on your PC.
# This script guides you through the exact steps.
# ============================================================

echo "=================================================="
echo "rclone Google Drive Setup — Lightning AI"
echo "=================================================="

# Fix config dir ownership (known Lightning bug)
echo "Fixing rclone config directory..."
if [ -d ~/.config/rclone ]; then
    OWNER=$(stat -c '%U' ~/.config/rclone)
    if [ "$OWNER" != "$USER" ]; then
        rmdir ~/.config/rclone
        mkdir -p ~/.config/rclone
        echo "✅ Fixed ownership"
    else
        echo "✅ Already correct"
    fi
else
    mkdir -p ~/.config/rclone
    echo "✅ Created"
fi

# Check if already configured
if rclone listremotes 2>/dev/null | grep -q "gdrive:"; then
    echo ""
    echo "✅ gdrive remote already configured!"
    echo "Testing access..."
    rclone lsd gdrive: --max-depth 1 2>/dev/null | head -5 && \
        echo "✅ Drive accessible" || echo "❌ Access failed — reconfigure below"
    echo ""
    read -p "Reconfigure anyway? (y/N): " RECONFIGURE
    [[ "$RECONFIGURE" != "y" ]] && exit 0
fi

echo ""
echo "=================================================="
echo "STEP 1: Run rclone config"
echo "=================================================="
echo ""
echo "When prompted, answer EXACTLY as follows:"
echo ""
echo "  > n          (new remote)"
echo "  > gdrive     (name)"
echo "  > drive      (type — Google Drive)"
echo "  > [Enter]    (client_id — leave blank)"
echo "  > [Enter]    (client_secret — leave blank)"
echo "  > 1          (scope — full drive access)"
echo "  > [Enter]    (root_folder_id — leave blank)"
echo "  > [Enter]    (service_account_file — leave blank)"
echo "  > n          (edit advanced config)"
echo "  > n          (use auto config) ← IMPORTANT"
echo ""
echo "  It will show: 'rclone authorize \"drive\"'"
echo "  Copy that EXACT command — you'll run it on your PC next."
echo ""
echo "Press Enter to start rclone config..."
read

rclone config

echo ""
echo "=================================================="
echo "STEP 2: Run authorize command on YOUR PC"
echo "=================================================="
echo ""
echo "On your Windows/Mac/Linux PC:"
echo ""
echo "  Windows: Download rclone from https://rclone.org/downloads/"
echo "           Extract zip → open folder → Shift+Right Click → 'Open PowerShell here'"
echo "           Run: .\\rclone.exe authorize \"drive\" \"eyJzY29wZSI6ImRyaXZlIn0\""
echo ""
echo "  Mac/Linux: brew install rclone OR curl https://rclone.org/install.sh | bash"
echo "             Then: rclone authorize \"drive\""
echo ""
echo "  This opens a browser → log in with Google → copy the JSON token shown"
echo "  Paste the token back into the waiting Lightning terminal"
echo ""
echo "=================================================="
echo "STEP 3: Verify"
echo "=================================================="
echo ""
echo "After config completes, run:"
echo "  rclone lsd gdrive: --max-depth 1"
echo ""
echo "You should see your Google Drive folders listed."
echo "Then run: bash lightning/download_dataset.sh"
