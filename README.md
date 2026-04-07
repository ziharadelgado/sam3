# SAM 3 Shark Segmentation Pipeline

Automated pipeline to convert bounding box annotations to segmentation masks using SAM 3 on Unity HPC.

## Workflow Overview

```
Google Drive (queue/) 
    ↓
Unity HPC (SAM 3 processing)
    ↓
Google Drive (processed/)
    ↓
Download & Upload to Roboflow
```

## Input Format

Your Google Drive `queue/` folder must contain:

```
queue/
├── annotations.json          # COCO format with bounding boxes
└── images/
    ├── 00001.jpg
    ├── 00002.jpg
    └── ...
```

**annotations.json structure (COCO format):**
```json
{
  "images": [
    {"id": 1, "file_name": "00001.jpg", "width": 1024, "height": 576}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, width, height]}
  ],
  "categories": [
    {"id": 1, "name": "shark"}
  ]
}
```

## Output Format

The pipeline produces YOLO segmentation format:

```
processed/
├── images/
│   ├── 00001.jpg
│   └── ...
├── labels/
│   ├── 00001.txt              # YOLO segmentation format
│   └── ...
└── data.yaml
```

**YOLO segmentation format (labels/*.txt):**
```
0 x1 y1 x2 y2 x3 y3 ...        # Normalized polygon coordinates
```

## Setup (One-time)

### 1. Clone Repository on Unity

```bash
cd /home/zihara_delgado_uri_edu/
git clone https://github.com/ziharadelgado/sam3-shark-pipeline.git
cd sam3-shark-pipeline
```

### 2. Create Conda Environment

Submit the setup job:

```bash
sbatch scripts/setup_env.slurm
```

Check progress:
```bash
tail -f logs/setup_*.out
```

### 3. Download SAM 3 Checkpoint

First, login to HuggingFace and request access:

```bash
huggingface-cli login
# Visit: https://huggingface.co/facebook/sam3 and request access
```

Then download:

```bash
bash scripts/download_sam3.sh
```

### 4. Configure rclone (if not already done)

```bash
rclone config
# Configure 'gdrive' remote for your Google Drive
```

## Running the Pipeline

### Method 1: Automatic (Recommended)

This syncs data and submits the job in one command:

```bash
bash scripts/sync_and_run.sh
```

### Method 2: Manual

Step-by-step if you need more control:

```bash
# 1. Sync queue manually
rclone copy gdrive:DeepSeaObjectDetection/rclone/queue/ ~/queue/ -v

# 2. Submit job
sbatch scripts/run_pipeline.slurm

# 3. Monitor
squeue -u $USER
tail -f logs/sam3_*.out
```

## Monitoring Job Progress

```bash
# Check job status
squeue -u zihara_delgado_uri_edu

# View live output
tail -f logs/sam3_<jobid>.out

# View errors (if any)
tail -f logs/sam3_<jobid>.err

# Cancel job if needed
scancel <jobid>
```

## After Processing

### 1. Download from Google Drive

The results are automatically uploaded to:
```
gdrive:DeepSeaObjectDetection/rclone/processed/
```

Download to your laptop:

```bash
# On your laptop (or Unity login node)
rclone copy gdrive:DeepSeaObjectDetection/rclone/processed/ ./local_processed/ -v
```

### 2. Upload to Roboflow

1. Go to your Roboflow project
2. Click "Upload Data"
3. Select format: **YOLO (Darknet) with segmentation**
4. Upload the `processed/` folder
5. The `data.yaml` will be detected automatically

## Project Structure

```
sam3-shark-pipeline/
├── scripts/
│   ├── setup_env.slurm          # One-time environment setup
│   ├── download_sam3.sh         # Download SAM 3 checkpoint
│   ├── run_pipeline.slurm       # Main processing job
│   └── sync_and_run.sh          # Wrapper script
├── src/
│   └── annotate_with_sam3.py    # Main annotation script
├── requirements.txt             # Python dependencies
├── logs/                        # SLURM job logs
└── README.md                    # This file
```

## Customization

### Change Text Prompt

Edit `scripts/run_pipeline.slurm` line 51:

```bash
--text-prompt "shark"     # Change to "fish", "Etmopterus", etc.
```

### Adjust Resources

Edit `scripts/run_pipeline.slurm` SBATCH headers:

```bash
#SBATCH --gpus=1                # Number of GPUs
#SBATCH --mem=60G               # RAM
#SBATCH --time=12:00:00         # Max runtime
```

### Change Directories

Edit paths in `scripts/run_pipeline.slurm`:

```bash
QUEUE_DIR="$WORK_DIR/queue"
OUTPUT_DIR="$WORK_DIR/processed"
GDRIVE_QUEUE="gdrive:DeepSeaObjectDetection/rclone/queue/"
GDRIVE_OUTPUT="gdrive:DeepSeaObjectDetection/rclone/processed/"
```

## Troubleshooting

### "SAM 3 checkpoint not found"

```bash
bash scripts/download_sam3.sh
```

### "No images found in queue"

Check your Google Drive structure:
```bash
rclone ls gdrive:DeepSeaObjectDetection/rclone/queue/
```

Should show:
```
annotations.json
images/00001.jpg
images/00002.jpg
...
```

### "CUDA out of memory"

Reduce batch size or use smaller images. Edit `annotate_with_sam3.py` if needed.

### "rclone sync failed"

Check rclone configuration:
```bash
rclone config show gdrive
```

Test connection:
```bash
rclone ls gdrive:
```

### Job stuck in queue

Check partition status:
```bash
sinfo
squeue
```

## Performance

- **L40S GPU (48GB VRAM)**: ~10-20 images/minute
- **SAM 3 model**: ~3.5GB VRAM
- **Processing time**: Depends on image count and size

## Contact

- **Author**: Zihara Delgado
- **Lab**: URI AI Lab (Professor Indrani Mandal)
- **Email**: zihara_delgado_uri_edu

## References

- SAM 3: [facebook/sam3](https://huggingface.co/facebook/sam3)
- Unity HPC: [URI Research Computing](https://web.uri.edu/hpc-research-computing/)
- YOLO Format: [Ultralytics Docs](https://docs.ultralytics.com/)
