#!/usr/bin/env python3
"""
Local test script for SAM 3 pipeline
Run this on your laptop to verify your annotations.json format before using Unity
"""

import json
import sys
from pathlib import Path

def validate_queue_folder(queue_dir):
    """Validate that queue folder has correct structure."""
    queue_path = Path(queue_dir)
    
    print("="*70)
    print("Validating Queue Folder")
    print("="*70)
    
    errors = []
    warnings = []
    
    # Check directory exists
    if not queue_path.exists():
        errors.append(f"Directory not found: {queue_dir}")
        return errors, warnings
    
    print(f"✓ Directory exists: {queue_path}")
    
    # Check annotations.json
    ann_file = queue_path / "annotations.json"
    if not ann_file.exists():
        errors.append("annotations.json not found")
    else:
        print(f"✓ Found annotations.json")
        
        # Validate JSON structure
        try:
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            # Check required keys
            required_keys = ['images', 'annotations', 'categories']
            for key in required_keys:
                if key not in data:
                    errors.append(f"Missing key in JSON: {key}")
                else:
                    print(f"  ✓ Has '{key}' ({len(data[key])} items)")
            
            if 'images' in data and 'annotations' in data:
                print(f"  - Images: {len(data['images'])}")
                print(f"  - Annotations: {len(data['annotations'])}")
                print(f"  - Categories: {len(data.get('categories', []))}")
                
                # Check image format
                if len(data['images']) > 0:
                    img = data['images'][0]
                    required_img_keys = ['id', 'file_name', 'width', 'height']
                    missing = [k for k in required_img_keys if k not in img]
                    if missing:
                        errors.append(f"Image entry missing keys: {missing}")
                
                # Check annotation format
                if len(data['annotations']) > 0:
                    ann = data['annotations'][0]
                    required_ann_keys = ['id', 'image_id', 'category_id', 'bbox']
                    missing = [k for k in required_ann_keys if k not in ann]
                    if missing:
                        errors.append(f"Annotation entry missing keys: {missing}")
                    
                    # Check bbox format
                    if 'bbox' in ann:
                        bbox = ann['bbox']
                        if not isinstance(bbox, list) or len(bbox) != 4:
                            errors.append(f"Invalid bbox format: {bbox} (expected [x,y,w,h])")
                
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")
        except Exception as e:
            errors.append(f"Error reading JSON: {e}")
    
    # Check images directory
    img_dir = queue_path / "images"
    if not img_dir.exists():
        errors.append("images/ directory not found")
    elif not img_dir.is_dir():
        errors.append("images/ is not a directory")
    else:
        # Count images
        image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        if len(image_files) == 0:
            warnings.append("No .jpg or .png files found in images/")
        else:
            print(f"✓ Found {len(image_files)} image files")
            
            # Sample filenames
            print(f"  First 5 images:")
            for img in sorted(image_files)[:5]:
                print(f"    - {img.name}")
    
    # Summary
    print("\n" + "="*70)
    if errors:
        print("❌ VALIDATION FAILED")
        print("="*70)
        for err in errors:
            print(f"  ERROR: {err}")
    else:
        print("✅ VALIDATION PASSED")
        print("="*70)
        print("Your queue folder is ready for processing!")
    
    if warnings:
        print("\nWarnings:")
        for warn in warnings:
            print(f"  ⚠️  {warn}")
    
    print("="*70)
    
    return errors, warnings


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_local.py <queue_directory>")
        print("")
        print("Example:")
        print("  python test_local.py /path/to/queue/")
        print("")
        print("Expected structure:")
        print("  queue/")
        print("  ├── annotations.json")
        print("  └── images/")
        print("      ├── 00001.jpg")
        print("      └── ...")
        sys.exit(1)
    
    queue_dir = sys.argv[1]
    errors, warnings = validate_queue_folder(queue_dir)
    
    if errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
