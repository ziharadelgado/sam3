# Quick Start Guide

## First Time Setup (10-15 minutes)

```bash
# 1. Clone repo on Unity
cd /home/zihara_delgado_uri_edu/
git clone https://github.com/ziharadelgado/sam3-shark-pipeline.git
cd sam3-shark-pipeline

# 2. Setup environment (submit job)
sbatch scripts/setup_env.slurm
# Wait ~30min, check: tail -f logs/setup_*.out

# 3. Login to HuggingFace
huggingface-cli login
# Get token from: https://huggingface.co/settings/tokens
# Request access: https://huggingface.co/facebook/sam3

# 4. Download SAM 3 checkpoint
bash scripts/download_sam3.sh
```

## Every Time You Want to Process Data

### Prepare Input

Put in Google Drive: `DeepSeaObjectDetection/rclone/queue/`
```
queue/
├── annotations.json     # COCO format bounding boxes
└── images/
    ├── 00001.jpg
    └── ...
```

### Run Pipeline

```bash
cd /home/zihara_delgado_uri_edu/sam3-shark-pipeline
bash scripts/sync_and_run.sh
```

### Monitor

```bash
# Check job
squeue -u $USER

# Watch progress
tail -f logs/sam3_*.out
```

### Download Results

Results auto-upload to: `DeepSeaObjectDetection/rclone/processed/`

Download locally:
```bash
rclone copy gdrive:DeepSeaObjectDetection/rclone/processed/ ./local_processed/ -v
```

### Upload to Roboflow

1. Roboflow → Upload Data
2. Format: **YOLO (Darknet) with segmentation**
3. Upload `processed/` folder
4. Done!

## Common Commands

```bash
# Check job status
squeue -u zihara_delgado_uri_edu

# Cancel job
scancel <jobid>

# View output
cat logs/sam3_<jobid>.out

# View errors
cat logs/sam3_<jobid>.err

# Check queue on Drive
rclone ls gdrive:DeepSeaObjectDetection/rclone/queue/

# Clean up queue (after processing)
rm -rf /home/zihara_delgado_uri_edu/queue/*
```

## Troubleshooting One-Liners

```bash
# Re-download checkpoint
bash scripts/download_sam3.sh

# Check conda env
conda env list | grep sam3

# Test imports
conda activate sam3
python -c "from sam3 import build_sam3_image_model; print('OK')"

# Test rclone
rclone ls gdrive:

# View all logs
ls -lht logs/
```
