#!/bin/bash
# Continuous checkpoint backup to S3 (runs every hour)
#
# USAGE:
#   chmod +x backup_checkpoints.sh
#   nohup ./backup_checkpoints.sh > backup.log 2>&1 &
#
# To stop: pkill -f backup_checkpoints.sh

set -euo pipefail

# Configuration
CHECKPOINT_DIR="/home/shadeform/checkpoints/reidnet_v3/work_dirs"
S3_BUCKET="data-labeling.livereachmedia.com"
S3_PREFIX="models/reidnet_v3/checkpoints"
BACKUP_INTERVAL=3600  # 1 hour in seconds

echo "[$(date)] 🚀 Starting checkpoint backup daemon"
echo "[$(date)] 📂 Local: ${CHECKPOINT_DIR}"
echo "[$(date)] ☁️  S3: s3://${S3_BUCKET}/${S3_PREFIX}"
echo "[$(date)] ⏱️  Interval: ${BACKUP_INTERVAL}s (1 hour)"
echo ""

while true; do
    # Check if checkpoint directory exists
    if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
        echo "[$(date)] ⚠️  Checkpoint directory not found, waiting..."
        sleep "${BACKUP_INTERVAL}"
        continue
    fi
    
    # Count checkpoint files
    CKPT_COUNT=$(find "${CHECKPOINT_DIR}" -name "*.pth" -o -name "*.pt" | wc -l)
    
    if [[ "${CKPT_COUNT}" -eq 0 ]]; then
        echo "[$(date)] ℹ️  No checkpoints found yet, waiting..."
    else
        echo "[$(date)] 📤 Backing up ${CKPT_COUNT} checkpoint(s)..."
        
        # Sync entire work_dirs to S3 (includes logs, configs, checkpoints)
        aws s3 sync "${CHECKPOINT_DIR}" "s3://${S3_BUCKET}/${S3_PREFIX}/" \
            --exclude "*.pyc" \
            --exclude "__pycache__/*" \
            --exclude ".git/*" \
            --quiet
        
        if [[ $? -eq 0 ]]; then
            echo "[$(date)] ✅ Backup complete"
        else
            echo "[$(date)] ❌ Backup failed (exit code: $?)"
        fi
    fi
    
    echo "[$(date)] 💤 Sleeping for 1 hour..."
    sleep "${BACKUP_INTERVAL}"
done
