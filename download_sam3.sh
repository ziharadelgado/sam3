#!/bin/bash
# Download SAM 3 checkpoint from HuggingFace
# Requires: HuggingFace access approved + huggingface-cli login

set -e

CHECKPOINT_DIR="/home/zihara_delgado_uri_edu/checkpoints"
CHECKPOINT_PATH="$CHECKPOINT_DIR/sam3.pt"

echo "========================================"
echo "SAM 3 Checkpoint Download"
echo "========================================"

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Check if already exists
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "✓ sam3.pt already exists at $CHECKPOINT_DIR"
    ls -lh "$CHECKPOINT_PATH"
    exit 0
fi

# Check HuggingFace login
echo ""
echo "Checking HuggingFace authentication..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "❌ ERROR: Not logged in to HuggingFace"
    echo ""
    echo "Please run:"
    echo "  huggingface-cli login"
    echo ""
    echo "Then request access at:"
    echo "  https://huggingface.co/facebook/sam3"
    exit 1
fi

echo "✓ Logged in to HuggingFace"

# Download checkpoint
echo ""
echo "Downloading SAM 3 checkpoint..."
echo "This may take several minutes..."

python3 << EOF
from huggingface_hub import hf_hub_download

try:
    hf_hub_download(
        repo_id='facebook/sam3',
        filename='sam3.pt',
        local_dir='$CHECKPOINT_DIR',
        resume_download=True
    )
    print('\n✓ SAM 3 checkpoint downloaded!')
except Exception as e:
    print(f'\n❌ Download failed: {e}')
    print('\nTroubleshooting:')
    print('1. Request access: https://huggingface.co/facebook/sam3')
    print('2. Check login: huggingface-cli whoami')
    print('3. Try re-login: huggingface-cli login')
    exit(1)
EOF

# Verify download
if [ -f "$CHECKPOINT_PATH" ]; then
    echo ""
    echo "========================================"
    echo "✅ Download Complete"
    echo "========================================"
    echo "Checkpoint saved to: $CHECKPOINT_PATH"
    ls -lh "$CHECKPOINT_PATH"
    echo "========================================"
else
    echo "❌ ERROR: Checkpoint file not found after download"
    exit 1
fi
