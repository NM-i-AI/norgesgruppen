# Fix PyTorch 2.6 weight loading BEFORE any ultralytics import
import torch

# Monkey-patch torch.load to disable weights_only restriction
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# Add safe globals for ultralytics weight loading
try:
    torch.serialization.add_safe_globals([
        'collections.OrderedDict',
        'torch.nn.modules.conv.Conv2d',
        'torch.nn.modules.batchnorm.BatchNorm2d',
        'torch.nn.modules.activation.SiLU',
        'torch.nn.modules.pooling.AdaptiveAvgPool2d',
        'torch.nn.modules.linear.Linear',
        'torch.nn.modules.dropout.Dropout',
        'ultralytics.nn.modules.conv.Conv',
        'ultralytics.nn.modules.block.C2f',
        'ultralytics.nn.modules.head.Detect'
    ])
except AttributeError:
    # Fallback for older PyTorch versions
    pass

import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import random
from utils import evaluate_predictions, load_coco_split

# Now safe to import ultralytics
from ultralytics import YOLO

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
                if single_class:
                    class_id = 0  # Force all annotations to class 0
                else:
                    class_id = ann['category_id']
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"Created YOLO labels in {output_dir} ({'single-class' if single_class else 'multi-class'})")

def create_yolo_dataset_files():
    """Create YOLO dataset files with proper train/val split."""
    data_path = Path("data")
    
    # Load splits
    train_coco = load_coco_split(data_path / "train_split.json")
    val_coco = load_coco_split(data_path / "val_split.json")
    
    # Get image filenames for each split
    train_images = [img['file_name'] for img in train_coco['images']]
    val_images = [img['file_name'] for img in val_coco['images']]
    
    # Create train.txt and val.txt files with image paths
    train_txt_path = data_path / "train.txt"
    val_txt_path = data_path / "val.txt"
    
    with open(train_txt_path, 'w') as f:
        for img_name in train_images:
            img_path = data_path / "train" / "images" / img_name
            f.write(f"{img_path.absolute()}\n")
    
    with open(val_txt_path, 'w') as f:
        for img_name in val_images:
            img_path = data_path / "train" / "images" / img_name
            f.write(f"{img_path.absolute()}\n")
    
    print(f"Created {train_txt_path} with {len(train_images)} images")
    print(f"Created {val_txt_path} with {len(val_images)} images")
    
    return train_txt_path, val_txt_path

def train_yolo_single_class():
    """Train YOLOv8x single-class detector at 1280px."""
    print("\n=== Training YOLOv8x Single-Class Detector ===")
    
    # Check if splits exist, create if needed
    data_path = Path("data")
    if not (data_path / "train_split.json").exists():
        print("Creating train/val splits...")
        create_train_val_split()
    
    # Create single-class labels directory
    labels_dir = data_path / "train" / "labels_single_class"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if single-class YOLO labels exist, create if needed
    if not list(labels_dir.glob("*.txt")):
        print("Creating single-class YOLO labels...")
        # Load full dataset and create single-class labels for all images
        with open(data_path / "train" / "annotations.json") as f:
            full_coco = json.load(f)
        
        coco_to_yolo_labels(full_coco, labels_dir, single_class=True)
    
    # Create train.txt and val.txt files
    train_txt_path, val_txt_path = create_yolo_dataset_files()
    
    # Create data.yaml with correct paths for single-class
    yaml_content = f"""# YOLO single-class dataset config
path: {data_path.absolute()}
train: {train_txt_path.name}
val: {val_txt_path.name}

nc: 1
names: ['product']
"""
    
    yaml_path = data_path / "data_single_class.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created {yaml_path}")
    
    try:
        # Initialize YOLOv8x model
        print("Initializing YOLOv8x model...")
        model = YOLO('yolov8x.pt')  # Large model for better detection
        
        print(f"Model loaded successfully")
        print(f"YOLOv8x summary: {model.model}")
        print(f"Model info: {model.info()}")
        
        # Determine device to use
        if torch.cuda.is_available():
            device = '0'  # Use first GPU
            print(f"Using CUDA device: {device}")
        else:
            device = 'cpu'
            print(f"CUDA not available, using CPU")
        
        # Train the model
        print("Starting training...")
        results = model.train(
            data=str(yaml_path),
            epochs=100,  # More epochs for single-class convergence
            batch=-1,    # Auto batch size
            imgsz=1280,  # High resolution for small products
            device=device,
            project='runs/detect',
            name='yolov8x_single_class_1280',
            exist_ok=True,
            verbose=True,
            save=True,
            plots=True,
            val=True,
            patience=30,  # More patience for larger model
            close_mosaic=50,  # Close mosaic at 50% of training
            amp=True,     # Mixed precision for memory efficiency
            cache=True    # Cache images for faster training
        )
        
        print(f"Training completed successfully!")
        print(f"Results: {results}")
        
        # Load validation split for evaluation
        val_coco = load_coco_split(data_path / "val_split.json")
        
        # Run inference on validation images
        print("\nRunning validation inference...")
        val_predictions = []
        
        val_image_dir = data_path / "train" / "images"
        val_image_ids = [img['id'] for img in val_coco['images']]
        
        for img_info in val_coco['images']:
            img_path = val_image_dir / img_info['file_name']
            
            if img_path.exists():
                # Run inference
                results = model(str(img_path), verbose=False, imgsz=1280)
                
                # Convert results to COCO format
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for i in range(len(boxes)):
                            # Extract box data
                            xyxy = boxes.xyxy[i].cpu().numpy()
                            conf = boxes.conf[i].cpu().numpy()
                            cls = int(boxes.cls[i].cpu().numpy())
                            
                            # Convert xyxy to xywh (COCO format)
                            x1, y1, x2, y2 = xyxy
                            x, y, w, h = x1, y1, x2-x1, y2-y1
                            
                            val_predictions.append({
                                'image_id': img_info['id'],
                                'category_id': cls,  # Will be 0 for single-class
                                'bbox': [float(x), float(y), float(w), float(h)],
                                'score': float(conf)
                            })
        
        print(f"Generated {len(val_predictions)} predictions")
        
        # Evaluate predictions - focus on detection metrics
        if val_predictions:
            val_score, det_map, cls_map = evaluate_predictions(val_predictions, val_coco)
            
            print(f"\n=== Validation Results ===")
            print(f"Detection mAP@0.5: {det_map:.4f}")
            print(f"Classification mAP@0.5: {cls_map:.4f}")
            print(f"Combined val_score: {val_score:.4f}")
            
            # Calculate recall metrics
            total_gt_boxes = len(val_coco['annotations'])
            print(f"Total ground truth boxes: {total_gt_boxes}")
            print(f"Total predictions: {len(val_predictions)}")
            
            # Report metrics for orchestrator
            print(f"\nMETRIC:val_score={val_score:.4f}")
            print(f"METRIC:detection_map={det_map:.4f}")
            print(f"METRIC:classification_map={cls_map:.4f}")
            print(f"METRIC:num_predictions={len(val_predictions)}")
            print(f"METRIC:num_gt_boxes={total_gt_boxes}")
            print(f"METRIC:model_size=yolov8x")
            print(f"METRIC:resolution=1280")
            print(f"METRIC:epochs=100")
            print(f"METRIC:single_class=1")
            
        else:
            print("No predictions generated - model may need more training")
            print(f"METRIC:val_score=0.0000")
            print(f"METRIC:detection_map=0.0000")
            print(f"METRIC:classification_map=0.0000")
            print(f"METRIC:num_predictions=0")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Report failure metrics
        print(f"METRIC:val_score=0.0000")
        print(f"METRIC:training_failed=1")
        raise

def main():
    """Main training pipeline."""
    print("=== YOLOv8x Single-Class Training Pipeline ===")
    
    # Train single-class detector
    train_yolo_single_class()
    
    print("\n=== Training Pipeline Complete ===")

if __name__ == "__main__":
    main()