#!/usr/bin/env python3
"""
Main experiment file. This is the entry point executed by the orchestrator.
Edit this file to implement experiments.

Print metrics to stdout as: METRIC:name=value
Example: METRIC:val_accuracy=0.74
"""

# MONKEY PATCH: Fix PyTorch 2.6 weights_only=True issue for ultralytics
import torch
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import random
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile
import yaml
from ultralytics import YOLO

def create_train_val_split(annotations_path, train_output_path, val_output_path, val_ratio=0.1, seed=42):
    """
    Split COCO annotations into train/val splits (90/10) by image, stratified by store section if possible.
    """
    print(f"Loading annotations from {annotations_path}...")
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Extract image info and try to stratify by store section (from filename pattern)
    images = coco_data['images']
    print(f"Total images: {len(images)}")
    
    # Group images by potential store section (extract from filename if possible)
    section_groups = defaultdict(list)
    for img in images:
        filename = img['file_name']
        # Try to extract section from filename pattern
        # Assuming filenames might contain section info, otherwise group all together
        section = 'unknown'  # Default section
        if 'egg' in filename.lower():
            section = 'Egg'
        elif 'frokost' in filename.lower():
            section = 'Frokost'
        elif 'knekkebrod' in filename.lower():
            section = 'Knekkebrod'
        elif 'varmedrikker' in filename.lower():
            section = 'Varmedrikker'
        
        section_groups[section].append(img)
    
    print(f"Images by section: {[(k, len(v)) for k, v in section_groups.items()]}")
    
    # Stratified split within each section
    random.seed(seed)
    np.random.seed(seed)
    
    val_images = []
    train_images = []
    
    for section, imgs in section_groups.items():
        imgs_copy = imgs.copy()
        random.shuffle(imgs_copy)
        
        n_val = max(1, int(len(imgs_copy) * val_ratio))  # At least 1 image for val if section exists
        section_val = imgs_copy[:n_val]
        section_train = imgs_copy[n_val:]
        
        val_images.extend(section_val)
        train_images.extend(section_train)
        
        print(f"Section {section}: {len(section_train)} train, {len(section_val)} val")
    
    print(f"Final split: {len(train_images)} train, {len(val_images)} val")
    
    # Get image IDs for filtering annotations
    train_img_ids = {img['id'] for img in train_images}
    val_img_ids = {img['id'] for img in val_images}
    
    # Filter annotations
    train_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in train_img_ids]
    val_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in val_img_ids]
    
    print(f"Annotations: {len(train_annotations)} train, {len(val_annotations)} val")
    
    # Create train split
    train_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': coco_data['categories'],
        'info': coco_data.get('info', {})
    }
    
    # Create val split
    val_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': coco_data['categories'],
        'info': coco_data.get('info', {})
    }
    
    # Save splits
    with open(train_output_path, 'w') as f:
        json.dump(train_data, f)
    print(f"Saved train split to {train_output_path}")
    
    with open(val_output_path, 'w') as f:
        json.dump(val_data, f)
    print(f"Saved val split to {val_output_path}")
    
    return train_data, val_data

def compute_val_score(gt_coco_path, pred_coco_path):
    """
    Compute val_score = 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5
    using pycocotools COCOeval.
    """
    # Load ground truth
    coco_gt = COCO(gt_coco_path)
    
    # Load predictions
    with open(pred_coco_path, 'r') as f:
        pred_data = json.load(f)
    
    if not pred_data:
        print("No predictions found")
        return 0.0, 0.0, 0.0
    
    coco_dt = coco_gt.loadRes(pred_data)
    
    # Detection mAP@0.5 (category agnostic)
    # Create detection-only predictions (all category_id = 1)
    det_pred_data = []
    for pred in pred_data:
        det_pred = pred.copy()
        det_pred['category_id'] = 1  # Single category for detection
        det_pred_data.append(det_pred)
    
    # Create detection-only ground truth
    det_gt_data = {
        'images': coco_gt.dataset['images'],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'product'}]
    }
    
    for ann in coco_gt.dataset['annotations']:
        det_ann = ann.copy()
        det_ann['category_id'] = 1
        det_gt_data['annotations'].append(det_ann)
    
    # Save temporary detection GT file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(det_gt_data, f)
        det_gt_path = f.name
    
    # Save temporary detection predictions file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(det_pred_data, f)
        det_pred_path = f.name
    
    try:
        # Compute detection mAP
        coco_det_gt = COCO(det_gt_path)
        coco_det_dt = coco_det_gt.loadRes(det_pred_path)
        
        coco_eval_det = COCOeval(coco_det_gt, coco_det_dt, 'bbox')
        coco_eval_det.params.iouThrs = [0.5]  # Only mAP@0.5
        coco_eval_det.evaluate()
        coco_eval_det.accumulate()
        coco_eval_det.summarize()
        
        detection_map = coco_eval_det.stats[1]  # mAP@0.5
        
        # Classification mAP@0.5 (category-aware)
        coco_eval_cls = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval_cls.params.iouThrs = [0.5]  # Only mAP@0.5
        coco_eval_cls.evaluate()
        coco_eval_cls.accumulate()
        coco_eval_cls.summarize()
        
        classification_map = coco_eval_cls.stats[1]  # mAP@0.5
        
        # Compute final score
        val_score = 0.7 * detection_map + 0.3 * classification_map
        
        print(f"Detection mAP@0.5: {detection_map:.4f}")
        print(f"Classification mAP@0.5: {classification_map:.4f}")
        print(f"Val Score: {val_score:.4f}")
        
        return val_score, detection_map, classification_map
        
    finally:
        # Clean up temporary files
        Path(det_gt_path).unlink(missing_ok=True)
        Path(det_pred_path).unlink(missing_ok=True)

def coco_to_yolo_format(coco_data, images_dir, output_dir, nc=356):
    """
    Convert COCO annotations to YOLO format.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create labels directory
    labels_dir = output_dir / 'labels'
    labels_dir.mkdir(exist_ok=True)
    
    # Create images symlink
    images_link = output_dir / 'images'
    if not images_link.exists():
        images_link.symlink_to(Path(images_dir).resolve())
    
    # Create image_id to annotations mapping
    img_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
    
    # Create image_id to image info mapping
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Convert each image's annotations
    for img_id, annotations in img_to_anns.items():
        img_info = img_id_to_info[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Get filename without extension for label file
        filename = Path(img_info['file_name']).stem
        label_file = labels_dir / f"{filename}.txt"
        
        with open(label_file, 'w') as f:
            for ann in annotations:
                # Convert COCO bbox [x, y, width, height] to YOLO [x_center, y_center, width, height] normalized
                x, y, w, h = ann['bbox']
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                
                # Category ID (YOLO uses 0-based indexing)
                if nc == 1:
                    # Single class for detection-only
                    class_id = 0
                else:
                    # Use original category ID (already 0-based in our case since we have category 0)
                    class_id = ann['category_id']
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    print(f"Converted {len(img_to_anns)} images to YOLO format in {output_dir}")
    return output_dir

def create_yolo_dataset_yaml(train_dir, val_dir, nc=356, output_path='dataset.yaml'):
    """
    Create YOLO dataset configuration YAML file.
    """
    # Create class names (simplified for now)
    if nc == 1:
        names = ['product']
    else:
        # Use category IDs as names for now (can be improved later with actual product names)
        names = [f'category_{i}' for i in range(nc + 1)]  # 0-356 = 357 classes
    
    dataset_config = {
        'path': str(Path.cwd().resolve()),  # Use absolute path
        'train': str(Path(train_dir).resolve()),  # Use absolute paths
        'val': str(Path(val_dir).resolve()),
        'nc': len(names),
        'names': names
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Created YOLO dataset config: {output_path}")
    return output_path

def yolo_predictions_to_coco(yolo_results, val_coco_path, output_path):
    """
    Convert YOLO predictions to COCO format for evaluation.
    """
    # Load validation COCO data to get image info
    with open(val_coco_path, 'r') as f:
        val_data = json.load(f)
    
    # Create mapping from image filename to image info
    filename_to_img = {}
    for img in val_data['images']:
        filename_to_img[img['file_name']] = img
    
    coco_predictions = []
    
    for result in yolo_results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
            
        # Get image info
        img_path = Path(result.path)
        img_filename = img_path.name
        
        if img_filename not in filename_to_img:
            print(f"Warning: {img_filename} not found in validation set")
            continue
            
        img_info = filename_to_img[img_filename]
        img_id = img_info['id']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Convert each detection
        boxes = result.boxes
        for i in range(len(boxes)):
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            
            # Convert to COCO format (x, y, width, height)
            x = float(x1)
            y = float(y1)
            width = float(x2 - x1)
            height = float(y2 - y1)
            
            # Get confidence score
            score = float(boxes.conf[i].cpu().numpy())
            
            # Get class ID (for nc=1, this will always be 0)
            class_id = int(boxes.cls[i].cpu().numpy())
            
            pred = {
                'image_id': img_id,
                'category_id': class_id,
                'bbox': [x, y, width, height],
                'score': score
            }
            coco_predictions.append(pred)
    
    # Save predictions
    with open(output_path, 'w') as f:
        json.dump(coco_predictions, f)
    
    print(f"Saved {len(coco_predictions)} predictions to {output_path}")
    return coco_predictions

def main():
    print("=== YOLOv8m nc=1 at 640px Training ===\n")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Paths
    annotations_path = 'data/train/annotations.json'
    train_split_path = 'data/train_split.json'
    val_split_path = 'data/val_split.json'
    images_dir = 'data/train/images'
    
    try:
        # Step 1: Create train/val split if not exists
        if not Path(train_split_path).exists() or not Path(val_split_path).exists():
            print("Creating train/val split...")
            train_data, val_data = create_train_val_split(
                annotations_path, train_split_path, val_split_path, val_ratio=0.1, seed=42
            )
        else:
            print("Loading existing train/val split...")
            with open(train_split_path, 'r') as f:
                train_data = json.load(f)
            with open(val_split_path, 'r') as f:
                val_data = json.load(f)
        
        # Step 2: Convert to YOLO format for nc=1
        print("\nConverting to YOLO format (nc=1)...")
        train_yolo_dir = coco_to_yolo_format(train_data, images_dir, 'data/yolo_train_nc1', nc=1)
        val_yolo_dir = coco_to_yolo_format(val_data, images_dir, 'data/yolo_val_nc1', nc=1)
        
        # Create dataset YAML for nc=1
        dataset_yaml = create_yolo_dataset_yaml(train_yolo_dir, val_yolo_dir, nc=1, output_path='dataset_nc1.yaml')
        
        # Step 3: Train YOLOv8m model at 640px
        print("\nTraining YOLOv8m model (nc=1, imgsz=640)...")
        model = YOLO('yolov8m.pt')  # Load pretrained YOLOv8m
        
        # Training parameters
        train_results = model.train(
            data=dataset_yaml,
            epochs=80,
            imgsz=640,
            batch=16,
            name='yolov8m_nc1_640px',
            project='runs/detect',
            save=True,
            val=True,
            plots=True,
            device=0 if Path('/proc/driver/nvidia/version').exists() else 'cpu',  # Use GPU if available
            workers=4,
            patience=20,
            close_mosaic=50,
            seed=42
        )
        
        print(f"Training completed. Best model saved to: {model.trainer.best}")
        
        # Step 4: Run validation on best model
        print("\nRunning validation...")
        best_model = YOLO(model.trainer.best)
        
        # Get validation image paths
        val_img_paths = []
        for img_info in val_data['images']:
            img_path = Path(images_dir) / img_info['file_name']
            val_img_paths.append(str(img_path))
        
        # Run inference on validation set
        val_results = best_model.predict(
            source=val_img_paths,
            conf=0.01,  # Low confidence threshold to maximize recall
            iou=0.7,    # NMS IoU threshold
            imgsz=640,
            save=False,
            verbose=False
        )
        
        # Convert predictions to COCO format
        pred_coco_path = 'yolo_nc1_640_predictions.json'
        yolo_predictions_to_coco(val_results, val_split_path, pred_coco_path)
        
        # Step 5: Evaluate using COCO metrics
        print("\nEvaluating with COCO metrics...")
        val_score, detection_map, classification_map = compute_val_score(val_split_path, pred_coco_path)
        
        # Report metrics
        print("\n=== Results ===\n")
        print(f"Detection mAP@0.5: {detection_map:.4f}")
        print(f"Classification mAP@0.5: {classification_map:.4f}")
        print(f"Val Score: {val_score:.4f}")
        
        # Output metrics for orchestrator
        print(f"METRIC:detection_map_50={detection_map:.4f}")
        print(f"METRIC:classification_map_50={classification_map:.4f}")
        print(f"METRIC:val_score={val_score:.4f}")
        print(f"METRIC:train_images={len(train_data['images'])}")
        print(f"METRIC:val_images={len(val_data['images'])}")
        print(f"METRIC:model_size=yolov8m")
        print(f"METRIC:input_size=640")
        print(f"METRIC:num_classes=1")
        print(f"METRIC:epochs=80")
        print(f"METRIC:batch_size=16")
        
        # Check if hypothesis is met (any non-zero detection score)
        hypothesis_met = detection_map > 0.0
        
        print(f"METRIC:hypothesis_met={'1.0' if hypothesis_met else '0.0'}")
        
        if hypothesis_met:
            print(f"\n✓ Hypothesis met: detection_mAP@0.5 = {detection_map:.4f} (> 0.0)")
        else:
            print(f"\n✗ Hypothesis not met: detection_mAP@0.5 = {detection_map:.4f} (= 0.0)")
            
        print(f"\n✓ YOLOv8m nc=1 at 640px training completed successfully!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("METRIC:val_score=0.0")
        print("METRIC:detection_map_50=0.0")
        print("METRIC:classification_map_50=0.0")
        print("METRIC:hypothesis_met=0.0")

if __name__ == "__main__":
    main()