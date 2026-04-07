# SAM 3 Pipeline Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GOOGLE DRIVE (Input)                             │
│  DeepSeaObjectDetection/rclone/queue/                               │
│    ├── annotations.json  (COCO format bounding boxes)               │
│    └── images/                                                       │
│        ├── 00001.jpg                                                 │
│        ├── 00002.jpg                                                 │
│        └── ...                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ rclone sync
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    UNITY HPC (Processing)                            │
│  /home/zihara_delgado_uri_edu/queue/                                │
│                                                                       │
│  1. Load SAM 3 model (facebook/sam3)                                │
│  2. For each image:                                                  │
│     - Read bounding boxes from annotations.json                      │
│     - Convert bbox → SAM 3 format                                    │
│     - Run SAM 3 with text prompt "shark"                            │
│     - Generate segmentation mask                                     │
│     - Convert mask → YOLO polygon format                            │
│  3. Export to YOLO segmentation format                              │
│                                                                       │
│  GPU: L40S (48GB VRAM)                                               │
│  Model size: ~3.5GB                                                  │
│  Speed: ~10-20 images/minute                                         │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ rclone sync
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   GOOGLE DRIVE (Output)                              │
│  DeepSeaObjectDetection/rclone/processed/                           │
│    ├── images/                                                       │
│    │   ├── 00001.jpg                                                 │
│    │   └── ...                                                       │
│    ├── labels/                                                       │
│    │   ├── 00001.txt  (YOLO segmentation)                          │
│    │   └── ...                                                       │
│    └── data.yaml                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ download locally
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ROBOFLOW                                        │
│  1. Upload Data                                                      │
│  2. Select format: YOLO (Darknet) with segmentation                │
│  3. Upload processed/ folder                                         │
│  4. Ready to train YOLO model!                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Format Transformations

### Input: COCO JSON (Bounding Boxes)
```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 180]  // [x, y, width, height]
    }
  ]
}
```

### Processing: SAM 3 Segmentation
```python
# bbox → SAM 3 format
sam3_bbox = [x, y, x+w, y+h]  # [100, 150, 300, 330]

# Run SAM 3 with text prompt
results = processor.set_text_prompt(state, prompt="shark")

# Get binary mask (H x W array)
mask = results[0]['segmentation']  # Shape: (576, 1024)
```

### Output: YOLO Segmentation Format
```
labels/00001.txt:
0 0.123 0.456 0.234 0.567 0.345 0.678 ...
│ └────────────────────────────────────┘
│              normalized polygon coords
└─ class_id (0 = shark)
```

## Command Summary

### One-time Setup
```bash
# 1. Clone repo
git clone https://github.com/ziharadelgado/sam3-shark-pipeline.git
cd sam3-shark-pipeline

# 2. Setup environment
sbatch scripts/setup_env.slurm

# 3. Download SAM 3
huggingface-cli login
bash scripts/download_sam3.sh
```

### Every Processing Run
```bash
# Single command
bash scripts/sync_and_run.sh

# Or step-by-step
rclone copy gdrive:DeepSeaObjectDetection/rclone/queue/ ~/queue/
sbatch scripts/run_pipeline.slurm
squeue -u $USER
```

### After Processing
```bash
# Download results
rclone copy gdrive:DeepSeaObjectDetection/rclone/processed/ ./results/

# Upload to Roboflow (manual UI upload)
```

## File Dependencies

```
SAM 3 Pipeline
├── Runtime Dependencies
│   ├── SAM 3 checkpoint (~3.5GB)
│   ├── PyTorch 2.7.0 + CUDA 12.6
│   ├── Python 3.12
│   └── rclone (Google Drive access)
│
├── Input Requirements
│   ├── annotations.json (COCO format)
│   ├── images/ (JPG or PNG)
│   └── Google Drive configured
│
└── Output Produced
    ├── images/ (copied from input)
    ├── labels/ (YOLO segmentation .txt)
    └── data.yaml (YOLO config)
```

## Error Recovery

### Job Failed?
```bash
# Check what went wrong
cat logs/sam3_<jobid>.err

# Common issues:
# - Queue empty → Check Google Drive
# - Checkpoint missing → bash scripts/download_sam3.sh
# - GPU OOM → Reduce batch size in annotate_with_sam3.py
```

### Partial Results?
```bash
# Results are in ~/processed/ even if upload fails
ls -lh ~/processed/images/ ~/processed/labels/

# Manual upload to Drive
rclone copy ~/processed/ gdrive:DeepSeaObjectDetection/rclone/processed/
```

### Need to Reprocess?
```bash
# Clean up and start fresh
rm -rf ~/queue/* ~/processed/*
bash scripts/sync_and_run.sh
```
