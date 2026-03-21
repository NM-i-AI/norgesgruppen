import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import random
from utils import evaluate_predictions

def create_train_val_split():
    """Create 90/10 stratified split by image, ensuring store sections are represented."""
    print("Creating train/val split...")
    
    data_path = Path("data")
    annotations_path = data_path / "train" / "annotations.json"
    
    with open(annotations_path) as f:
        coco_data = json.load(f)
    
    # Identify store sections from filenames
    store_sections = defaultdict(list)
    section_patterns = {
        'Egg': ['egg'],
        'Frokost': ['frokost'], 
        'Knekkebrod': ['knekkebrod'],
        'Varmedrikker': ['varmedrikker']
    }
    
    for img in coco_data['images']:
        fname = img['file_name'].lower()
        assigned = False
        for section, patterns in section_patterns.items():
            if any(pattern in fname for pattern in patterns):
                store_sections[section].append(img['id'])
                assigned = True
                break
        if not assigned:
            store_sections['unknown'].append(img['id'])
    
    print(f"Store section distribution:")
    for section, img_ids in store_sections.items():
        print(f"  {section}: {len(img_ids)} images")
    
    # Stratified split - 10% from each section for validation
    random.seed(42)
    val_image_ids = set()
    train_image_ids = set()
    
    for section, img_ids in store_sections.items():
        img_ids_copy = img_ids.copy()
        random.shuffle(img_ids_copy)
        
        val_count = max(1, len(img_ids_copy) // 10)  # At least 1 image per section
        val_ids = img_ids_copy[:val_count]
        train_ids = img_ids_copy[val_count:]
        
        val_image_ids.update(val_ids)
        train_image_ids.update(train_ids)
        
        print(f"  {section}: {len(train_ids)} train, {len(val_ids)} val")
    
    print(f"\nTotal split: {len(train_image_ids)} train, {len(val_image_ids)} val")
    
    # Create train split COCO data
    train_images = [img for img in coco_data['images'] if img['id'] in train_image_ids]
    train_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in train_image_ids]
    
    train_coco = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': coco_data['categories'],
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', [])
    }
    
    # Create val split COCO data
    val_images = [img for img in coco_data['images'] if img['id'] in val_image_ids]
    val_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in val_image_ids]
    
    val_coco = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': coco_data['categories'],
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', [])
    }
    
    # Save split files
    with open(data_path / "train_split.json", 'w') as f:
        json.dump(train_coco, f)
    
    with open(data_path / "val_split.json", 'w') as f:
        json.dump(val_coco, f)
    
    print(f"\nSaved train_split.json ({len(train_annotations)} annotations)")
    print(f"Saved val_split.json ({len(val_annotations)} annotations)")
    
    return train_coco, val_coco

def coco_to_yolo_labels(coco_data, output_dir, single_class=False):
    """Convert COCO annotations to YOLO format labels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create image_id to filename mapping
    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    id_to_size = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
    
    # Group annotations by image
    image_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_annotations[ann['image_id']].append(ann)
    
    # Convert each image's annotations
    for image_id, annotations in image_annotations.items():
        filename = id_to_filename[image_id]
        img_w, img_h = id_to_size[image_id]
        
        # Create label filename (replace .jpg with .txt)
        label_filename = Path(filename).stem + '.txt'
        label_path = output_dir / label_filename
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # Convert COCO bbox [x, y, width, height] to YOLO [x_center, y_center, width, height] normalized
                x, y, w, h = ann['bbox']
                x_center = (x + w/2) / img_w
                y_center = (y + h/2) / img_h
                width = w / img_w
                height = h / img_h
                
                # Class ID (0 for single class, original category_id for multi-class)
                class_id = 0 if single_class else ann['category_id']
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"Created YOLO labels in {output_dir} ({'single-class' if single_class else 'multi-class'})")

def create_data_yaml(split_type, num_classes):
    """Create data.yaml file for ultralytics training."""
    data_path = Path("data")
    
    # Create class names list
    if num_classes == 1:
        names = ['product']
    else:
        # Load category names from original annotations
        with open(data_path / "train" / "annotations.json") as f:
            coco_data = json.load(f)
        
        # Create names list indexed by category_id
        names = [''] * 357  # 0-356
        for cat in coco_data['categories']:
            names[cat['id']] = cat['name']
    
    yaml_content = f"""# YOLO dataset config
path: {data_path.absolute()}
train: train/images
val: train/images  # We'll filter by split during training

nc: {num_classes}
names: {names}
"""
    
    yaml_filename = f"data_{split_type}.yaml"
    with open(data_path / yaml_filename, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created {yaml_filename}")
    return data_path / yaml_filename

def test_evaluation_function():
    """Test the evaluation function with dummy predictions."""
    print("\nTesting evaluation function...")
    
    # Load val split for testing
    data_path = Path("data")
    with open(data_path / "val_split.json") as f:
        val_coco = json.load(f)
    
    # Create dummy predictions
    dummy_predictions = []
    
    # Add some correct predictions (high scores)
    for i, ann in enumerate(val_coco['annotations'][:10]):
        # Perfect detection
        dummy_predictions.append({
            'image_id': ann['image_id'],
            'category_id': ann['category_id'],
            'bbox': ann['bbox'],
            'score': 0.9
        })
        
        # Detection-only (wrong category)
        if i < 5:
            dummy_predictions.append({
                'image_id': ann['image_id'], 
                'category_id': 0,  # Wrong category
                'bbox': ann['bbox'],
                'score': 0.8
            })
    
    # Add some false positives
    for img in val_coco['images'][:3]:
        dummy_predictions.append({
            'image_id': img['id'],
            'category_id': 1,
            'bbox': [10, 10, 50, 50],  # Random box
            'score': 0.7
        })
    
    # Test evaluation
    try:
        val_score, det_map, cls_map = evaluate_predictions(dummy_predictions, val_coco)
        print(f"Evaluation test successful!")
        print(f"  Detection mAP@0.5: {det_map:.4f}")
        print(f"  Classification mAP@0.5: {cls_map:.4f}")
        print(f"  Combined val_score: {val_score:.4f}")
        
        # Verify score calculation
        expected_score = 0.7 * det_map + 0.3 * cls_map
        print(f"  Expected score: {expected_score:.4f} (matches: {abs(val_score - expected_score) < 1e-6})")
        
    except Exception as e:
        print(f"Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main data preparation pipeline."""
    print("=== Data Preparation Pipeline ===")
    
    # Step 1: Create train/val split
    train_coco, val_coco = create_train_val_split()
    
    # Step 2: Create YOLO labels for multi-class (356 categories)
    data_path = Path("data")
    labels_mc_dir = data_path / "labels_mc"
    coco_to_yolo_labels(train_coco, labels_mc_dir / "train", single_class=False)
    coco_to_yolo_labels(val_coco, labels_mc_dir / "val", single_class=False)
    
    # Step 3: Create YOLO labels for single-class (detection only)
    labels_sc_dir = data_path / "labels_sc"
    coco_to_yolo_labels(train_coco, labels_sc_dir / "train", single_class=True)
    coco_to_yolo_labels(val_coco, labels_sc_dir / "val", single_class=True)
    
    # Step 4: Create data.yaml files
    yaml_mc = create_data_yaml("mc", 357)  # 0-356 categories
    yaml_sc = create_data_yaml("sc", 1)    # Single class
    
    # Step 5: Test evaluation function
    test_evaluation_function()
    
    print("\n=== Data Preparation Complete ===")
    print(f"Files created:")
    print(f"  - data/train_split.json ({len(train_coco['annotations'])} annotations)")
    print(f"  - data/val_split.json ({len(val_coco['annotations'])} annotations)")
    print(f"  - data/labels_mc/train/ and data/labels_mc/val/ (multi-class YOLO labels)")
    print(f"  - data/labels_sc/train/ and data/labels_sc/val/ (single-class YOLO labels)")
    print(f"  - data/data_mc.yaml and data/data_sc.yaml (ultralytics configs)")
    
    # Report metrics for orchestrator
    print(f"\nMETRIC:train_images={len(train_coco['images'])}")
    print(f"METRIC:val_images={len(val_coco['images'])}")
    print(f"METRIC:train_annotations={len(train_coco['annotations'])}")
    print(f"METRIC:val_annotations={len(val_coco['annotations'])}")
    print(f"METRIC:num_categories={len(train_coco['categories'])}")

if __name__ == "__main__":
    main()