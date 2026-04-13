#!/usr/bin/env python3
"""
SAM 3 Shark Annotation Pipeline
Reads bounding boxes from COCO JSON, converts to segmentation masks using SAM 3


KEY FEATURES:
1. Inverted mask detection and correction
   - SAM 3 sometimes marks BACKGROUND (water) instead of FOREGROUND (shark)
   - Detects this by checking if mask overlaps crop borders (which are always water)
   
2. Bbox border exclusion
   - Ensures segmentation NEVER includes pixels where bbox border would be drawn
   - Prevents overlap with yellow bbox line in Roboflow
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
import gc

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
        self.processor = Sam3Processor(sam3_model, confidence_threshold=0.1)
        
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
    
    def fix_inverted_mask(self, mask, cropped_image):
        """
        Detect and fix inverted masks (when SAM 3 marks background instead of shark).
        
        Strategy: Sample border pixels (which are almost always water/background).
        If mask heavily overlaps borders, it's inverted.
        
        This is a common issue with SAM models on low-contrast subjects:
        - Bioluminescent sharks have low contrast with dark water
        - SAM 3 sometimes thinks the shark is "background" and water is "foreground"
        - We detect this by checking border coverage
        
        Args:
            mask: Binary mask (bool or uint8)
            cropped_image: PIL Image (cropped bbox region)
        
        Returns:
            Corrected mask (bool)
        """
        crop_height, crop_width = mask.shape
        
        # Create border mask (10px band around edges)
        # This band is almost always water/background (95%+ of the time)
        border_mask = np.zeros_like(mask, dtype=bool)
        border_width = 10
        
        # Mark border pixels (top, bottom, left, right)
        border_mask[:border_width, :] = True  # Top edge
        border_mask[-border_width:, :] = True  # Bottom edge
        border_mask[:, :border_width] = True  # Left edge
        border_mask[:, -border_width:] = True  # Right edge
        
        # Convert mask to bool if needed
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        # Calculate how much of the border is marked as "shark"
        border_pixels_marked = (mask & border_mask).sum()
        total_border_pixels = border_mask.sum()
        border_coverage = border_pixels_marked / total_border_pixels
        
        print(f"  Border analysis: {border_coverage:.1%} of border marked as shark")
        
        # If >50% of border is marked as "shark", the mask is INVERTED
        # (SAM 3 is marking water as shark, and shark as water)
        if border_coverage > 0.5:
            print(f"  🔄 INVERTING mask (was marking water, now marks shark)")
            mask = ~mask  # Bitwise NOT: flip all pixels (0→1, 1→0)
        else:
            print(f"  ✓ Mask orientation correct (shark is foreground)")
        
        return mask
    
    def exclude_bbox_border_pixels(self, mask, crop_width, crop_height, border_thickness=2):
        """
        Exclude pixels at the crop edges where the bbox border would be.
        
        Since we crop with 10px padding, the original bbox border is approximately
        at the edge of the crop. We exclude these edge pixels to ensure the 
        segmentation never overlaps with where the bbox line would be drawn.
        
        The bbox SURROUNDS the shark but NEVER touches it, so pixels at the
        crop border are ALWAYS water/background, never shark.
        
        Visual explanation:
        ╔═══════════════════╗  ← These 2-3 pixels = FORBIDDEN (bbox line zone)
        ║░░░░░░░░░░░░░░░░░░░║  
        ║░┌───────────────┐░║  ← Original bbox (where yellow line would be)
        ║░│   🦈 shark    │░║  ← Only INSIDE can have segmentation
        ║░└───────────────┘░║
        ║░░░░░░░░░░░░░░░░░░░║
        ╚═══════════════════╝  ← FORBIDDEN
        
        Args:
            mask: Binary mask (bool or uint8)
            crop_width: Width of cropped image
            crop_height: Height of cropped image  
            border_thickness: Pixels to exclude from each edge (2-3 recommended)
        
        Returns:
            Mask with border zone excluded (bool)
        """
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        # Create mask for "forbidden zone" (where bbox border would be drawn)
        forbidden_zone = np.zeros((crop_height, crop_width), dtype=bool)
        
        # Mark the outer edges as forbidden
        forbidden_zone[:border_thickness, :] = True              # Top
        forbidden_zone[-border_thickness:, :] = True             # Bottom
        forbidden_zone[:, :border_thickness] = True              # Left
        forbidden_zone[:, -border_thickness:] = True             # Right
        
        # Remove forbidden pixels from segmentation
        mask_cleaned = mask & (~forbidden_zone)
        
        excluded = (mask & forbidden_zone).sum()
        if excluded > 0:
            print(f"  🚫 Removed {excluded} pixels from bbox border zone")
        
        return mask_cleaned
    
    def process_image_with_boxes(self, image_path, bboxes):
        """
        Process single image with bounding boxes using SAM 3.
        
        Uses CROPPING strategy: crops image to bbox before SAM 3,
        ensuring SAM 3 can ONLY segment within the bbox region.
        
        Includes two critical fixes:
        1. Inverted mask detection (water vs shark confusion)
        2. Bbox border exclusion (prevents overlap with bbox line)
        
        Args:
            image_path: Path to image
            bboxes: List of COCO format bboxes [x, y, w, h]
            
        Returns:
            List of segmentation masks (one per bbox)
        """
        # CRITICAL: Clear GPU cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Load FULL image once
        full_image = Image.open(image_path).convert('RGB')
        full_width, full_height = full_image.size
        
        masks = []
        
        # Process each bbox INDEPENDENTLY with CROPPING
        for idx, bbox in enumerate(bboxes):
            try:
                # Extract bbox coordinates
                x, y, w, h = bbox
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Add small padding to give SAM 3 context (10 pixels)
                # This helps SAM 3 see the boundary between shark and water
                padding = 10
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(full_width - x_pad, w + 2*padding)
                h_pad = min(full_height - y_pad, h + 2*padding)
                
                print(f"\n  Bbox {idx+1}/{len(bboxes)}")
                print(f"  Original bbox: [{x}, {y}, {w}, {h}]")
                print(f"  Padded crop: [{x_pad}, {y_pad}, {w_pad}, {h_pad}]")
                
                # CROP image to bbox region (with padding)
                cropped_image = full_image.crop((x_pad, y_pad, x_pad + w_pad, y_pad + h_pad))
                crop_width, crop_height = cropped_image.size
                
                print(f"  Cropped image size: {crop_width}x{crop_height}")
                
                # Set the CROPPED image in SAM 3
                inference_state = self.processor.set_image(cropped_image)
                
                # Create bbox that covers the ENTIRE cropped image
                # (since we already cropped to the bbox, just segment everything)
                # Format: [center_x, center_y, width, height] normalized [0,1]
                full_crop_box = [0.5, 0.5, 1.0, 1.0]  # Center of crop, full size
                
                print(f"  SAM 3 prompt: segment entire crop [0.5, 0.5, 1.0, 1.0]")
                
                # Add prompt to segment the entire cropped region
                inference_state = self.processor.add_geometric_prompt(
                    state=inference_state,
                    box=full_crop_box,
                    label=True  # Positive prompt
                )
                
                # Extract masks from inference_state
                if 'masks' in inference_state and len(inference_state['masks']) > 0:
                    # Get the first (best) mask from CROPPED image
                    crop_mask = inference_state['masks'][0].cpu().numpy()
                    
                    # Convert to binary mask
                    if crop_mask.ndim == 3:
                        crop_mask = crop_mask[0]
                    
                    crop_binary_mask = (crop_mask > 0).astype(np.uint8)
                    
                    # Check if mask is reasonable
                    crop_coverage = crop_binary_mask.sum() / (crop_width * crop_height)
                    print(f"  Crop mask coverage: {crop_coverage:.2%}")
                    
                    if crop_coverage < 0.001:
                        print(f"  ⚠️ WARNING: Mask too small - using bbox fallback!")
                        fallback_mask = np.zeros((full_height, full_width), dtype=np.uint8)
                        fallback_mask[y:y+h, x:x+w] = 1
                        masks.append(fallback_mask.astype(bool))
                        del inference_state
                        continue
                    
                    # CRITICAL FIX 1: Check if mask is INVERTED
                    # (SAM 3 marking water instead of shark)
                    crop_binary_mask_bool = crop_binary_mask.astype(bool)
                    crop_binary_mask_bool = self.fix_inverted_mask(
                        crop_binary_mask_bool, 
                        cropped_image
                    )
                    
                    # CRITICAL FIX 2: Exclude pixels where bbox border would be drawn
                    # (prevents segmentation from overlapping with yellow bbox line)
                    crop_binary_mask_bool = self.exclude_bbox_border_pixels(
                        crop_binary_mask_bool,
                        crop_width,
                        crop_height,
                        border_thickness=2  # 2-3px is safe
                    )
                    
                    crop_binary_mask = crop_binary_mask_bool.astype(np.uint8)
                    
                    # Clean the cropped mask (remove small disconnected regions)
                    cleaned_crop_mask = self.clean_mask(crop_binary_mask)
                    
                    # CRITICAL: Map the cropped mask back to FULL image coordinates
                    full_mask = np.zeros((full_height, full_width), dtype=np.uint8)
                    
                    # Place the cropped mask in the correct position
                    full_mask[y_pad:y_pad+crop_height, x_pad:x_pad+crop_width] = cleaned_crop_mask
                    
                    masks.append(full_mask.astype(bool))
                    print(f"  ✓ Mask generated and mapped to full image")
                    
                else:
                    # Fallback: create mask from bbox
                    print(f"  ⚠️ SAM 3 returned no mask - using bbox as fallback")
                    fallback_mask = np.zeros((full_height, full_width), dtype=np.uint8)
                    fallback_mask[y:y+h, x:x+w] = 1
                    masks.append(fallback_mask.astype(bool))
                
                # Clean up
                del inference_state
                    
            except Exception as e:
                print(f"  ❌ Error processing bbox {bbox}: {e}")
                import traceback
                traceback.print_exc()
                # Fallback mask from bbox
                fallback_mask = np.zeros((full_height, full_width), dtype=np.uint8)
                x, y, w, h = [int(v) for v in bbox]
                fallback_mask[y:y+h, x:x+w] = 1
                masks.append(fallback_mask.astype(bool))
        
        # CRITICAL: Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
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
    
    # Process ALL images (with and without annotations)
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
        
        # Copy image (ALWAYS - all frames)
        shutil.copy(img_path, output_images / filename)
        
        # Get bounding boxes for this image
        if img_id not in annotations_by_image:
            # No annotations for this image - create empty label file
            label_path = output_labels / f"{img_path.stem}.txt"
            label_path.touch()
            processed_count += 1
            continue
        
        bboxes = [ann['bbox'] for ann in annotations_by_image[img_id]]
        
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"Bboxes: {len(bboxes)}")
        print(f"{'='*60}")
        
        # CRITICAL: Create FRESH annotator for EACH image
        try:
            # Create new annotator instance
            annotator = SAM3SharkAnnotator(checkpoint_path)
            
            # Process this image with CROPPING + INVERSION detection + BORDER exclusion
            masks = annotator.process_image_with_boxes(img_path, bboxes)
            
            # CRITICAL: Delete annotator immediately after use
            del annotator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Export masks to YOLO segmentation format
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            label_path = output_labels / f"{img_path.stem}.txt"
            with open(label_path, 'w') as f:
                for mask in masks:
                    # Use a temporary annotator just for conversion
                    temp_annotator = SAM3SharkAnnotator(checkpoint_path)
                    yolo_seg = temp_annotator.mask_to_yolo_segmentation(
                        mask, img_width, img_height
                    )
                    del temp_annotator
                    
                    if yolo_seg:
                        # Class 0 for shark
                        f.write(f"0 {yolo_seg}\n")
            
            processed_count += 1
            annotated_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
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
