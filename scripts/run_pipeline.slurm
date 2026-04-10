#!/bin/bash
#SBATCH --job-name=sam3_shark_seg
#SBATCH --output=logs/sam3_%j.out
#SBATCH --error=logs/sam3_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00

# SAM 3 Shark Segmentation Pipeline
# Reads bounding boxes from queue, converts to segmentation with SAM 3

set -e  # Exit on error

echo "========================================"
echo "SAM 3 Shark Segmentation Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "========================================"

# 1. Environment Setup
module purge
module load cuda/12.6
module load conda/latest
conda activate sam3

# Create log directory
mkdir -p logs

# 2. Paths
WORK_DIR="/home/zihara_delgado_uri_edu"
QUEUE_DIR="$WORK_DIR/queue"
OUTPUT_DIR="$WORK_DIR/processed"
CHECKPOINT="$WORK_DIR/checkpoints/sam3.pt"
GDRIVE_QUEUE="gdrive:DeepSea_ObjectDetection/rclone/queue/"
GDRIVE_OUTPUT="gdrive:DeepSea_ObjectDetection/rclone/processed/"

# 3. Download queue from Google Drive
echo ""
echo "Step 1: Downloading queue from Google Drive..."
mkdir -p "$QUEUE_DIR"

timeout 1200s rclone copy "$GDRIVE_QUEUE" "$QUEUE_DIR" -v --transfers 8

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to download queue from Google Drive"
    exit 1
fi

# Unzip if there's a zip file in queue
if ls "$QUEUE_DIR"/*.zip 1> /dev/null 2>&1; then
    echo "Found zip file, extracting..."
    cd "$QUEUE_DIR"
    unzip -o *.zip
    cd -
    echo "✓ Extraction complete"
fi

# Move files from subdirectories to queue root (but skip images/ folder)
for subdir in "$QUEUE_DIR"/*/ ; do
    if [ -d "$subdir" ]; then
        dirname=$(basename "$subdir")
        if [ "$dirname" != "images" ]; then
            echo "Moving files from subdirectory: $dirname"
            mv "$subdir"/* "$QUEUE_DIR/" 2>/dev/null || true
            rmdir "$subdir" 2>/dev/null || true
        fi
    fi
done

# Check that we have data
if [ ! -f "$QUEUE_DIR/annotations.json" ]; then
    echo "❌ ERROR: annotations.json not found in queue"
    exit 1
fi

if [ ! -d "$QUEUE_DIR/images" ] || [ -z "$(ls -A $QUEUE_DIR/images)" ]; then
    echo "❌ ERROR: No images found in queue/images/"
    exit 1
fi

num_images=$(ls "$QUEUE_DIR/images" | wc -l)
echo "✓ Found annotations.json and $num_images images"

# 4. Check SAM 3 checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ ERROR: SAM 3 checkpoint not found at $CHECKPOINT"
    echo "Run: bash scripts/download_sam3.sh"
    exit 1
fi

echo "✓ SAM 3 checkpoint found"

# 5. Run SAM 3 Processing
echo ""
echo "Step 2: Running SAM 3 segmentation..."
mkdir -p "$OUTPUT_DIR"

cd /home/zihara_delgado_uri_edu/sam3-pipeline

python3 -u src/annotate_with_sam3.py \
    --queue-dir "$QUEUE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: SAM 3 processing failed"
    exit 1
fi

# 6. Upload results to Google Drive
echo ""
echo "Step 3: Uploading results to Google Drive..."

timeout 1800s rclone copy "$OUTPUT_DIR" "$GDRIVE_OUTPUT" -v --transfers 8

if [ $? -ne 0 ]; then
    echo "⚠️  WARNING: Upload to Google Drive failed or timed out"
    echo "Results are still available at: $OUTPUT_DIR"
else
    echo "✓ Results uploaded to Google Drive"
fi

# 7. Summary
echo ""
echo "========================================"
echo "✅ Pipeline Complete"
echo "========================================"
echo "Input: $QUEUE_DIR"
echo "Output: $OUTPUT_DIR"
echo "Google Drive: $GDRIVE_OUTPUT"
echo "Finished: $(date)"
echo "========================================"

# Optional: Clean up queue to save space (commented out by default)
# rm -rf "$QUEUE_DIR"
