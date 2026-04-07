#!/usr/bin/env python3
"""
SAM 3 Shark Annotation Pipeline
Reads bounding boxes from COCO JSON, converts to segmentation masks using SAM 3
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

# SAM 3 imports
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualize.utils import draw_box_and_masks


class SAM3SharkAnnotator:
    def __init__(self, checkpoint_path, device='cuda'):
        """Initialize SAM 3 model for shark segmentation."""
        print(f"Loading SAM 3 from {checkpoint_path}...")
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Build SAM 3 model
        sam3_model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        self.processor = Sam3Processor(sam3_model)
        
        print(f"✓ SAM 3 loaded on {self.device}")
        
    def bbox_to_sam3_format(self, bbox):
        """Convert COCO bbox [x, y, w, h] to SAM 3 format [x1, y1, x2, y2]."""
        x, y, w, h = bbox
        return [x, y, x + w, y + h]
    
    def process_image_with_boxes(self, image_path, bboxes, text_prompt="shark"):
        """
        Process single image with bounding boxes using SAM 3.
        
        Args:
            image_path: Path to image
            bboxes: List of COCO format bboxes [x, y, w, h]
            text_prompt: Text prompt for SAM 3 (default: "shark")
            
        Returns:
            List of segmentation masks (one per bbox)
        """
        image = Image.open(image_path).convert('RGB')
        
        # Set image in processor
        state = self.processor.set_image(image)
        
        masks = []
        for bbox in bboxes:
            # Convert bbox to SAM 3 format
            sam3_bbox = self.bbox_to_sam3_format(bbox)
            
            # Run SAM 3 with text prompt
            # SAM 3 uses text prompts to understand what to segment
            results = self.processor.set_text_prompt(
                state=state,
                prompt=text_prompt
            )
            
            # Extract mask from results
            if results and len(results) > 0:
                # Get the best mask
                mask = results[0]['segmentation']
                masks.append(mask)
            else:
                # Fallback: create mask from bbox if SAM 3 fails
                h, w = np.array(image).shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                x, y, bw, bh = bbox
                mask[int(y):int(y+bh), int(x):int(x+bw)] = 1
                masks.append(mask)
        
        return masks
    
    def mask_to_polygon(self, mask):
        """Convert binary mask to polygon coordinates for YOLO format."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Approximate polygon
        epsilon = 0.002 * cv2.arcLength(largest, True)
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
    checkpoint_path,
    text_prompt="shark"
):
    """
    Process entire queue folder: read COCO annotations, run SAM 3, export YOLO format.
    
    Args:
        queue_dir: Input folder with images/ and annotations.json
        output_dir: Output folder for processed data
        checkpoint_path: Path to SAM 3 checkpoint
        text_prompt: Text prompt for SAM 3
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
    
    print(f"Found {len(image_map)} images with {len(coco_data['annotations'])} annotations")
    
    # Initialize SAM 3
    annotator = SAM3SharkAnnotator(checkpoint_path)
    
    # Process each image
    processed_count = 0
    skipped_count = 0
    
    for img_id, img_info in tqdm(image_map.items(), desc="Processing images"):
        filename = img_info['file_name']
        img_path = images_dir / filename
        
        if not img_path.exists():
            print(f"⚠️  Image not found: {filename}")
            skipped_count += 1
            continue
        
        # Get bounding boxes for this image
        if img_id not in annotations_by_image:
            # No annotations for this image, copy as-is
            shutil.copy(img_path, output_images / filename)
            # Create empty label file
            label_path = output_labels / f"{img_path.stem}.txt"
            label_path.touch()
            continue
        
        bboxes = [ann['bbox'] for ann in annotations_by_image[img_id]]
        
        # Run SAM 3 on this image
        try:
            masks = annotator.process_image_with_boxes(
                img_path,
                bboxes,
                text_prompt=text_prompt
            )
            
            # Copy image
            shutil.copy(img_path, output_images / filename)
            
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
    print(f"Processed: {processed_count} images")
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
    parser.add_argument(
        '--text-prompt',
        type=str,
        default='shark',
        help='Text prompt for SAM 3 (default: shark)'
    )
    
    args = parser.parse_args()
    
    # Run processing
    process_queue_folder(
        queue_dir=args.queue_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        text_prompt=args.text_prompt
    )


if __name__ == '__main__':
    main()
