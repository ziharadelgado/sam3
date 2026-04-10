#!/usr/bin/env python3
"""
SAM 3 Shark Annotation Pipeline
Reads bounding boxes from COCO JSON, converts to segmentation masks using SAM 3
Based on official SAM 3 examples and Kamron's SAM 2 pipeline
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import cv2
from tqdm import tqdm

# SAM 3 imports (based on official example)
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox


class SAM3SharkAnnotator:
    def __init__(self, checkpoint_path, device='cuda'):
        """Initialize SAM 3 model for shark segmentation."""
        print(f"Loading SAM 3 from {checkpoint_path}...")
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Enable optimizations for Ampere GPUs
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Use bfloat16 for efficiency
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        
        # Build SAM 3 model
        sam3_model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        self.processor = Sam3Processor(sam3_model, confidence_threshold=0.3)
        
        print(f"✓ SAM 3 loaded on {self.device}")
    
    def clean_mask(self, mask):
        """
        Remove small disconnected regions from mask.
        Keeps only the largest connected component (shark body).
        Removes bite marks and other small artifacts.
        
        Based on Kamron's clean_mask function from SAM 2 pipeline.
        """
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find all connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, 
            connectivity=8
        )
        
        # Keep only the largest component (excluding background label 0)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned = (labels == largest_label).astype(np.uint8)
        else:
            cleaned = mask_uint8
        
        return cleaned.astype(bool)
    
    def process_image_with_boxes(self, image_path, bboxes):
        """
        Process single image with bounding boxes using SAM 3.
        
        Args:
            image_path: Path to image
            bboxes: List of COCO format bboxes [x, y, w, h]
            
        Returns:
            List of segmentation masks (one per bbox)
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        
        # Set image in processor
        inference_state = self.processor.set_image(image)
        
        masks = []
        
        for bbox in bboxes:
            try:
                # Convert COCO bbox [x,y,w,h] to tensor
                box_xywh = torch.tensor([bbox], dtype=torch.float32)
                
                # Convert to CXCYWH format (center_x, center_y, width, height)
                box_cxcywh = box_xywh_to_cxcywh(box_xywh)
                
                # Normalize bbox to [0,1]
                norm_box = normalize_bbox(box_cxcywh, width, height).flatten().tolist()
                
                # Reset prompts for new prediction
                self.processor.reset_all_prompts(inference_state)
                
                # Add bounding box prompt
                # SAM 3 API: add_geometric_prompt() internally calls _forward_grounding()
                # which populates state["masks"], state["boxes"], state["scores"]
                inference_state = self.processor.add_geometric_prompt(
                    state=inference_state,
                    box=norm_box,
                    label=True  # Positive prompt
                )
                
                # Extract masks from inference_state
                # SAM 3 does NOT have a .predict() method
                # Results are stored directly in the state after add_geometric_prompt()
                if 'masks' in inference_state and len(inference_state['masks']) > 0:
                    # Get the first (best) mask
                    mask = inference_state['masks'][0].cpu().numpy()
                    
                    # Convert to binary mask
                    if mask.ndim == 3:
                        mask = mask[0]  # Take first channel if needed
                    
                    # CRITICAL: Clean mask to remove bite marks and noise
                    binary_mask = (mask > 0).astype(np.uint8)
                    cleaned_mask = self.clean_mask(binary_mask)
                    
                    masks.append(cleaned_mask)
                else:
                    # Fallback: create mask from bbox
                    print(f"  ⚠️  SAM 3 returned no mask for bbox, using bbox as fallback")
                    fallback_mask = np.zeros((height, width), dtype=np.uint8)
                    x, y, w, h = [int(v) for v in bbox]
                    fallback_mask[y:y+h, x:x+w] = 1
                    masks.append(fallback_mask.astype(bool))
                    
            except Exception as e:
                print(f"  ❌ Error processing bbox {bbox}: {e}")
                # Fallback mask from bbox
                fallback_mask = np.zeros((height, width), dtype=np.uint8)
                x, y, w, h = [int(v) for v in bbox]
                fallback_mask[y:y+h, x:x+w] = 1
                masks.append(fallback_mask.astype(bool))
        
        return masks
    
    def mask_to_polygon(self, mask):
        """
        Convert binary mask to polygon coordinates for YOLO format.
        
        Uses CHAIN_APPROX_NONE to preserve all contour points,
        then applies light smoothing. Based on Kamron's approach.
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE  # Keep all points for detailed contour
        )
        
        if not contours:
            return None
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Apply light smoothing (reduced from 0.002 to 0.001)
        epsilon = 0.001 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        
        return approx.flatten().tolist()
    
    def mask_to_yolo_segmentation(self, mask, img_width, img_height):
        """
        Convert mask to YOLO segmentation format (normalized polygon).
        
        Returns:
            String of normalized coordinates: "x1 y1 x2 y2 x3 y3 ..."
        """
        polygon = self.mask_to_polygon(mask)
        
        if polygon is None or len(polygon) < 6:
            return None
        
        # Normalize coordinates
        normalized = []
        for i in range(0, len(polygon), 2):
            x = polygon[i] / img_width
            y = polygon[i+1] / img_height
            normalized.extend([x, y])
        
        return ' '.join(map(str, normalized))


def process_queue_folder(
    queue_dir,
    output_dir,
    checkpoint_path
):
    """
    Process entire queue folder: read COCO annotations, run SAM 3, export YOLO format.
    
    Args:
        queue_dir: Input folder with images/ and annotations.json
        output_dir: Output folder for processed data
        checkpoint_path: Path to SAM 3 checkpoint
    """
    queue_path = Path(queue_dir)
    output_path = Path(output_dir)
    
    # Check inputs
    images_dir = queue_path / "images"
    annotations_file = queue_path / "annotations.json"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    
    # Create output structure
    output_images = output_path / "images"
    output_labels = output_path / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    # Load COCO annotations
    print(f"\nReading annotations from {annotations_file}...")
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Build image_id -> filename mapping
    image_map = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    print(f"Found {len(image_map)} total images")
    print(f"Found {len(annotations_by_image)} images WITH annotations")
    print(f"Found {len(coco_data['annotations'])} total bounding boxes")
    
    # Initialize SAM 3
    annotator = SAM3SharkAnnotator(checkpoint_path)
    
    # Process each image
    processed_count = 0
    skipped_count = 0
    annotated_count = 0
    
    for img_id, img_info in tqdm(image_map.items(), desc="Processing images"):
        filename = img_info['file_name']
        img_path = images_dir / filename
        
        if not img_path.exists():
            print(f"⚠️  Image not found: {filename}")
            skipped_count += 1
            continue
        
        # Copy image
        shutil.copy(img_path, output_images / filename)
        
        # Get bounding boxes for this image
        if img_id not in annotations_by_image:
            # No annotations for this image - create empty label file
            label_path = output_labels / f"{img_path.stem}.txt"
            label_path.touch()
            processed_count += 1
            continue
        
        bboxes = [ann['bbox'] for ann in annotations_by_image[img_id]]
        
        # Run SAM 3 on this image
        try:
            masks = annotator.process_image_with_boxes(img_path, bboxes)
            
            # Export masks to YOLO segmentation format
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            label_path = output_labels / f"{img_path.stem}.txt"
            with open(label_path, 'w') as f:
                for mask in masks:
                    yolo_seg = annotator.mask_to_yolo_segmentation(
                        mask, img_width, img_height
                    )
                    if yolo_seg:
                        # Class 0 for shark
                        f.write(f"0 {yolo_seg}\n")
            
            processed_count += 1
            annotated_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            skipped_count += 1
    
    # Create data.yaml for YOLO
    data_yaml = output_path / "data.yaml"
    with open(data_yaml, 'w') as f:
        f.write(f"""# SAM 3 Shark Dataset
train: images/
val: images/

nc: 1
names: ['shark']
""")
    
    print(f"\n{'='*70}")
    print(f"✅ PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total images: {len(image_map)}")
    print(f"Images WITH annotations: {annotated_count}")
    print(f"Images WITHOUT annotations (empty labels): {processed_count - annotated_count}")
    print(f"Skipped: {skipped_count} images")
    print(f"Output directory: {output_path}")
    print(f"Format: YOLO segmentation")
    print(f"{'='*70}\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3 Shark Annotation Pipeline"
    )
    parser.add_argument(
        '--queue-dir',
        type=str,
        required=True,
        help='Input queue directory with images/ and annotations.json'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for processed annotations'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/home/zihara_delgado_uri_edu/checkpoints/sam3.pt',
        help='Path to SAM 3 checkpoint'
    )
    
    args = parser.parse_args()
    
    # Run processing
    process_queue_folder(
        queue_dir=args.queue_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint
    )


if __name__ == '__main__':
    main()
