#!/bin/bash
# Sync queue from Google Drive and submit SLURM job
# Run this on Unity login node: bash scripts/sync_and_run.sh

set -e

PROJECT_DIR="/home/zihara_delgado_uri_edu/sam3-pipeline"
QUEUE_DIR="/home/zihara_delgado_uri_edu/queue"
GDRIVE_QUEUE="gdrive:DeepSea_ObjectDetection/rclone/queue/"

echo "========================================"
echo "SAM 3 Shark Pipeline - Sync & Submit"
echo "========================================"

# 1. Check that project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ ERROR: Project directory not found: $PROJECT_DIR"
    echo "Please clone the repository first"
    exit 1
fi

# 2. Sync queue from Google Drive (on login node)
echo ""
echo "Step 1: Syncing queue from Google Drive..."
mkdir -p "$QUEUE_DIR"

rclone copy "$GDRIVE_QUEUE" "$QUEUE_DIR" -v --transfers 8

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to sync queue from Google Drive"
    echo "Check rclone configuration: rclone config"
    exit 1
fi

# Verify we have data
if [ ! -f "$QUEUE_DIR/annotations.json" ]; then
    echo "❌ ERROR: annotations.json not found in queue"
    exit 1
fi

if [ ! -d "$QUEUE_DIR/images" ] || [ -z "$(ls -A $QUEUE_DIR/images)" ]; then
    echo "❌ ERROR: No images found in queue/images/"
    exit 1
fi

num_images=$(ls "$QUEUE_DIR/images" | wc -l)
num_annotations=$(grep -o '"image_id"' "$QUEUE_DIR/annotations.json" | wc -l)

echo "✓ Sync complete"
echo "  - Images: $num_images"
echo "  - Annotations: $num_annotations"

# 3. Submit SLURM job
echo ""
echo "Step 2: Submitting SLURM job..."

cd "$PROJECT_DIR"
job_id=$(sbatch scripts/run_pipeline.slurm | grep -oP '\d+')

if [ -n "$job_id" ]; then
    echo "✓ Job submitted: $job_id"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u $USER"
    echo "  tail -f logs/sam3_${job_id}.out"
else
    echo "❌ ERROR: Failed to submit job"
    exit 1
fi

echo ""
echo "========================================"
echo "Next steps:"
echo "1. Wait for job to complete"
echo "2. Download from Google Drive: processed/"
echo "3. Upload to Roboflow"
echo "========================================"
