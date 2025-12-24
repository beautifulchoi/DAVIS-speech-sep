#!/bin/bash
# ===========================================
# Script: run_extract_frames.sh
# Purpose: Run frame extraction script with predefined paths
# ===========================================

# === User-configurable variables ===
VIDEO_PATH="/home/prj/data/AVE_Dataset/AVE"       # Directory containing input videos
OUT_DIR="/home/prj/data/AVE_Dataset/preprocess/frames"          # Output directory for extracted frames

# === Optional: Python environment ===
# Uncomment and modify if you use conda or venv
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# === Run the extraction ===
echo "==========================================="
echo " Running Frame Extraction Script"
echo "-------------------------------------------"
echo " Video Directory : $VIDEO_PATH"
echo " Output Directory: $OUT_DIR"
echo "==========================================="

python extract_frames.py \
    --video_path "$VIDEO_PATH" \
    --out_dir "$OUT_DIR"

# === Check exit status ===
if [ $? -eq 0 ]; then
    echo "✅ Frame extraction completed successfully!"
else
    echo "❌ Frame extraction encountered an error!"
fi

