#!/usr/bin/env python3
"""
SAM 3 Shark Annotation Pipeline
Reads bounding boxes from COCO JSON, converts to segmentation masks using SAM 3
 
 
KEY FEATURES:
1. Smart mask inversion detection
   - Pixels touching the bbox border are ALWAYS water/background
   - If the largest component touches borders heavily → it's water → invert
   - After inversion, keep the largest interior component → that's the shark
 
2. Bbox border exclusion
   - Ensures segmentation NEVER includes pixels where bbox border would be drawn
   - Prevents overlap with yellow bbox line in Roboflow
 
3. Single model load
   - SAM 3 model is loaded ONCE and reused for all images
   - Prevents GPU memory leak from repeated model loading
 
4. Coverage filters
   - Masks covering less than 5% of the crop are discarded (probably noise)
   - Masks covering more than 95% are inverted (probably marking water)
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
        """
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, 
            connectivity=8
        )
        
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned = (labels == largest_label).astype(np.uint8)
        else:
            cleaned = mask_uint8
        
        return cleaned.astype(bool)
    
    def get_border_mask(self, height, width, border_width=5):
        """
        Create a mask of pixels that touch the crop border.
        These pixels are ALWAYS water/background, never shark.
        
        The shark is INSIDE the bounding box. The bbox border
        surrounds the shark. So pixels at the edge of the crop
        (where the bbox line is) are guaranteed to be water.
        """
        border = np.zeros((height, width), dtype=bool)
        border[:border_width, :] = True   # Top
        border[-border_width:, :] = True  # Bottom
        border[:, :border_width] = True   # Left
        border[:, -border_width:] = True  # Right
        return border
 
    def fix_inverted_mask(self, mask, crop_height, crop_width):
        """
        Detect and fix inverted masks using border-touching logic.
        
        RULE: Pixels touching the bbox border are ALWAYS water.
        The shark is always INSIDE, not touching the edges.
        
        Strategy:
        1. Get the largest connected component
        2. Check how much of it touches the crop border
        3. If it touches the border heavily → it's water → invert the mask
        4. After inversion, clean again to get the real shark
        """
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        # Step 1: Get largest component
        mask_cleaned = self.clean_mask(mask)
        
        # Step 2: Create border mask (5px band around edges)
        border_mask = self.get_border_mask(crop_height, crop_width, border_width=5)
        
        # Step 3: How much of the largest component touches the border?
        component_pixels = mask_cleaned.sum()
        if component_pixels == 0:
            print(f"  ⚠️ Empty mask after cleaning")
            return mask_cleaned
        
        border_touching = (mask_cleaned & border_mask).sum()
        
        # What fraction of the border is covered by the mask?
        total_border_pixels = border_mask.sum()
        border_coverage = border_touching / total_border_pixels if total_border_pixels > 0 else 0
        
        # What fraction of the mask is touching the border?
        border_ratio = border_touching / component_pixels
        
        print(f"  Border analysis: {border_ratio:.1%} of mask touches border, "
              f"{border_coverage:.1%} of border is covered by mask")
        
        # DECISION: If the mask covers >30% of the border pixels,
        # it's almost certainly water (water fills edges, shark doesn't)
        if border_coverage > 0.30:
            print(f"  🔄 INVERTING mask (largest component is water, not shark)")
            mask = ~mask
            # After inversion, clean again to get the shark
            mask = self.clean_mask(mask)
            
            # Verify the inverted mask is reasonable
            new_coverage = mask.sum() / (crop_height * crop_width)
            if new_coverage < 0.01:
                print(f"  ⚠️ Inverted mask too small ({new_coverage:.1%}), reverting")
                mask = mask_cleaned  # Revert to original
            else:
                print(f"  ✓ After inversion: {new_coverage:.1%} coverage (shark)")
        else:
            print(f"  ✓ Mask is correct (shark is foreground, doesn't touch borders)")
            mask = mask_cleaned
        
        return mask
    
    def exclude_bbox_border_pixels(self, mask, crop_width, crop_height, border_thickness=3):
        """
        Exclude pixels at the crop edges where the bbox border would be.
        Prevents segmentation from overlapping with yellow bbox line in Roboflow.
        """
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        forbidden_zone = np.zeros((crop_height, crop_width), dtype=bool)
        forbidden_zone[:border_thickness, :] = True
        forbidden_zone[-border_thickness:, :] = True
        forbidden_zone[:, :border_thickness] = True
        forbidden_zone[:, -border_thickness:] = True
        
        mask_cleaned = mask & (~forbidden_zone)
        
        excluded = (mask & forbidden_zone).sum()
        if excluded > 0:
            print(f"  🚫 Removed {excluded} pixels from bbox border zone")
        
        return mask_cleaned
    
    def process_image_with_boxes(self, image_path, bboxes):
        """
        Process single image with bounding boxes using SAM 3.
        
        Pipeline for each bbox:
        1. Crop image to bbox with 30px padding (more context for SAM 3)
        2. Run SAM 3 on the crop
        3. Smart inversion: if largest component touches borders → it's water → invert
        4. Exclude bbox border pixels
        5. Coverage filters: too small (<5%) or too large (>95%) → bbox fallback
        6. Map mask back to full image coordinates
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        full_image = Image.open(image_path).convert('RGB')
        full_width, full_height = full_image.size
        
        masks = []
        
        for idx, bbox in enumerate(bboxes):
            try:
                x, y, w, h = bbox
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # 30px padding gives SAM 3 more context to distinguish
                # shark from water, especially near nets/bites
                padding = 30
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(full_width - x_pad, w + 2*padding)
                h_pad = min(full_height - y_pad, h + 2*padding)
                
                print(f"\n  Bbox {idx+1}/{len(bboxes)}")
                print(f"  Original bbox: [{x}, {y}, {w}, {h}]")
                print(f"  Padded crop: [{x_pad}, {y_pad}, {w_pad}, {h_pad}]")
                
                cropped_image = full_image.crop((x_pad, y_pad, x_pad + w_pad, y_pad + h_pad))
                crop_width, crop_height = cropped_image.size
                
                print(f"  Cropped image size: {crop_width}x{crop_height}")
                
                # Run SAM 3
                inference_state = self.processor.set_image(cropped_image)
                
                full_crop_box = [0.5, 0.5, 1.0, 1.0]
                print(f"  SAM 3 prompt: segment entire crop [0.5, 0.5, 1.0, 1.0]")
                
                inference_state = self.processor.add_geometric_prompt(
                    state=inference_state,
                    box=full_crop_box,
                    label=True
                )
                
                if 'masks' in inference_state and len(inference_state['masks']) > 0:
                    crop_mask = inference_state['masks'][0].cpu().numpy()
                    
                    if crop_mask.ndim == 3:
                        crop_mask = crop_mask[0]
                    
                    crop_binary_mask = (crop_mask > 0).astype(bool)
                    
                    raw_coverage = crop_binary_mask.sum() / (crop_width * crop_height)
                    print(f"  Raw mask coverage: {raw_coverage:.2%}")
                    
                    # Empty mask → fallback
                    if raw_coverage < 0.001:
                        print(f"  ⚠️ WARNING: Mask empty - using bbox fallback!")
                        fallback_mask = np.zeros((full_height, full_width), dtype=np.uint8)
                        fallback_mask[y:y+h, x:x+w] = 1
                        masks.append(fallback_mask.astype(bool))
                        del inference_state
                        continue
                    
                    # SMART INVERSION: border-touching detection
                    crop_binary_mask = self.fix_inverted_mask(
                        crop_binary_mask,
                        crop_height,
                        crop_width
                    )
                    
                    # Exclude bbox border pixels
                    crop_binary_mask = self.exclude_bbox_border_pixels(
                        crop_binary_mask,
                        crop_width,
                        crop_height,
                        border_thickness=3
                    )
                    
                    # Final coverage check
                    final_coverage = crop_binary_mask.sum() / (crop_width * crop_height)
                    print(f"  Final mask coverage: {final_coverage:.2%}")
                    
                    # Too small → probably noise, not the shark
                    if final_coverage < 0.05:
                        print(f"  ⚠️ WARNING: Mask too small ({final_coverage:.1%}%) "
                              f"- using bbox fallback!")
                        fallback_mask = np.zeros((full_height, full_width), dtype=np.uint8)
                        fallback_mask[y:y+h, x:x+w] = 1
                        masks.append(fallback_mask.astype(bool))
                        del inference_state
                        continue
                    
                    # Too large → still marking water
                    if final_coverage > 0.95:
                        print(f"  ⚠️ WARNING: Mask covers {final_coverage:.1%}% "
                              f"- probably water, using bbox fallback!")
                        fallback_mask = np.zeros((full_height, full_width), dtype=np.uint8)
                        fallback_mask[y:y+h, x:x+w] = 1
                        masks.append(fallback_mask.astype(bool))
                        del inference_state
                        continue
                    
                    # Map mask back to full image coordinates
                    full_mask = np.zeros((full_height, full_width), dtype=np.uint8)
                    full_mask[y_pad:y_pad+crop_height, x_pad:x_pad+crop_width] = \
                        crop_binary_mask.astype(np.uint8)
                    
                    masks.append(full_mask.astype(bool))
                    print(f"  ✓ Mask generated and mapped to full image")
                    
                else:
                    print(f"  ⚠️ SAM 3 returned no mask - using bbox as fallback")
                    fallback_mask = np.zeros((full_height, full_width), dtype=np.uint8)
                    fallback_mask[y:y+h, x:x+w] = 1
                    masks.append(fallback_mask.astype(bool))
                
                del inference_state
                    
            except Exception as e:
                print(f"  ❌ Error processing bbox {bbox}: {e}")
                import traceback
                traceback.print_exc()
                fallback_mask = np.zeros((full_height, full_width), dtype=np.uint8)
                x, y, w, h = [int(v) for v in bbox]
                fallback_mask[y:y+h, x:x+w] = 1
                masks.append(fallback_mask.astype(bool))
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return masks
    
    def mask_to_polygon(self, mask):
        """Convert binary mask to polygon coordinates for YOLO format."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.001 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        
        return approx.flatten().tolist()
    
    def mask_to_yolo_segmentation(self, mask, img_width, img_height):
        """Convert mask to YOLO segmentation format (normalized polygon)."""
        polygon = self.mask_to_polygon(mask)
        
        if polygon is None or len(polygon) < 6:
            return None
        
        normalized = []
        for i in range(0, len(polygon), 2):
            x = polygon[i] / img_width
            y = polygon[i+1] / img_height
            normalized.extend([x, y])
        
        return ' '.join(map(str, normalized))
 
 
def process_queue_folder(queue_dir, output_dir, checkpoint_path):
    """
    Process entire queue folder: read COCO annotations, run SAM 3, export YOLO format.
    """
    queue_path = Path(queue_dir)
    output_path = Path(output_dir)
    
    images_dir = queue_path / "images"
    annotations_file = queue_path / "annotations.json"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    
    output_images = output_path / "images"
    output_labels = output_path / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    print(f"\nReading annotations from {annotations_file}...")
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    image_map = {img['id']: img for img in coco_data['images']}
    
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    print(f"Found {len(image_map)} total images")
    print(f"Found {len(annotations_by_image)} images WITH annotations")
    print(f"Found {len(coco_data['annotations'])} total bounding boxes")
    
    # Load SAM 3 model ONCE
    annotator = SAM3SharkAnnotator(checkpoint_path)
    
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
        
        shutil.copy(img_path, output_images / filename)
        
        if img_id not in annotations_by_image:
            label_path = output_labels / f"{img_path.stem}.txt"
            label_path.touch()
            processed_count += 1
            continue
        
        bboxes = [ann['bbox'] for ann in annotations_by_image[img_id]]
        
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"Bboxes: {len(bboxes)}")
        print(f"{'='*60}")
        
        try:
            masks = annotator.process_image_with_boxes(img_path, bboxes)
            
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            label_path = output_labels / f"{img_path.stem}.txt"
            with open(label_path, 'w') as f:
                for mask in masks:
                    yolo_seg = annotator.mask_to_yolo_segmentation(
                        mask, img_width, img_height
                    )
                    if yolo_seg:
                        f.write(f"0 {yolo_seg}\n")
            
            processed_count += 1
            annotated_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            skipped_count += 1
    
    del annotator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
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
        '--queue-dir', type=str, required=True,
        help='Input queue directory with images/ and annotations.json'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Output directory for processed annotations'
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default='/home/zihara_delgado_uri_edu/checkpoints/sam3.pt',
        help='Path to SAM 3 checkpoint'
    )
    
    args = parser.parse_args()
    
    process_queue_folder(
        queue_dir=args.queue_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint
    )
 
 
if __name__ == '__main__':
    main()
