import subprocess
import sys

# Install required packages at runtime
try:
    import pycocotools
except ImportError:
    print("Installing pycocotools...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pycocotools'], check=True)
    import pycocotools

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics==8.1.0'], check=True)
    from ultralytics import YOLO

import json
import pathlib
import shutil
from collections import defaultdict
import random
import cv2
import numpy as np
import tempfile
import torch

# Import required packages
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Set torch to use weights_only=False to avoid the unpickler issue
torch.serialization.DEFAULT_PROTOCOL = 2

def convert_coco_to_yolo_multiclass(coco_ann_file, images_dir, output_dir, train_ratio=0.8):
    """Convert COCO annotations to YOLO format preserving all 357 categories (0-356)"""
    
    with open(coco_ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    output_path = pathlib.Path(output_dir)
    train_images_dir = output_path / 'train' / 'images'
    train_labels_dir = output_path / 'train' / 'labels'
    val_images_dir = output_path / 'val' / 'images'
    val_labels_dir = output_path / 'val' / 'labels'
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    # Create image_id to filename mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Split images into train/val
    image_ids = list(image_info.keys())
    random.shuffle(image_ids)
    split_idx = int(len(image_ids) * train_ratio)
    train_image_ids = image_ids[:split_idx]
    val_image_ids = image_ids[split_idx:]
    
    # Get category info
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    num_categories = len(categories)
    
    print(f"Train images: {len(train_image_ids)}, Val images: {len(val_image_ids)}")
    print(f"Multi-class detection: {num_categories} categories (0-{max(categories.keys())})")
    
    def process_split(image_ids, images_dir_out, labels_dir_out, split_name):
        for image_id in image_ids:
            img_info = image_info[image_id]
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Copy image
            src_img_path = pathlib.Path(images_dir) / img_filename
            dst_img_path = images_dir_out / img_filename
            if src_img_path.exists():
                shutil.copy2(src_img_path, dst_img_path)
            else:
                print(f"Warning: Image {src_img_path} not found")
                continue
            
            # Convert annotations to YOLO format
            yolo_annotations = []
            for ann in annotations_by_image[image_id]:
                bbox = ann['bbox']  # [x, y, width, height] in pixels
                x, y, w, h = bbox
                
                # Convert to YOLO format: [class_id, x_center, y_center, width, height] normalized
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                
                # MULTI-CLASS: Keep original category_id
                class_id = ann['category_id']
                
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
            
            # Write YOLO label file
            label_filename = img_filename.replace('.jpg', '.txt')
            label_path = labels_dir_out / label_filename
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
    
    process_split(train_image_ids, train_images_dir, train_labels_dir, 'train')
    process_split(val_image_ids, val_images_dir, val_labels_dir, 'val')
    
    # Create dataset.yaml for all categories
    # Create class names list (0-356)
    max_cat_id = max(categories.keys())
    class_names = ['unknown'] * (max_cat_id + 1)  # Initialize with 'unknown'
    for cat_id, cat_name in categories.items():
        class_names[cat_id] = cat_name
    
    dataset_yaml = f"""path: {output_path.absolute()}
train: train/images
val: val/images

nc: {len(class_names)}
names: {class_names}
"""
    
    with open(output_path / 'dataset.yaml', 'w') as f:
        f.write(dataset_yaml)
    
    return train_image_ids, val_image_ids, categories

def train_yolo_model(dataset_yaml_path, epochs=50, imgsz=1280):
    """Train YOLOv8m model for multi-class detection with basic augmentations"""
    # Use weights_only=False to avoid the unpickler issue
    model = YOLO('yolov8m.pt', task='detect')  # Load pretrained YOLOv8m model
    
    # Train the model with basic augmentations
    results = model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=6,  # Batch size as requested
        device=0,  # Use GPU
        project='runs/detect',
        name='yolov8m_multiclass',
        save=True,
        verbose=True,
        # Basic training hyperparameters
        lr0=0.01,        # Initial learning rate
        lrf=0.01,        # Final learning rate
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=20,     # Early stopping patience
        # Basic augmentations
        mosaic=1.0,      # Mosaic augmentation probability
        mixup=0.0,       # No MixUp for simplicity
        copy_paste=0.0,  # No copy-paste for simplicity
        # Standard augmentations
        hsv_h=0.015,     # HSV-Hue augmentation
        hsv_s=0.7,       # HSV-Saturation augmentation
        hsv_v=0.4,       # HSV-Value augmentation
        degrees=0.0,     # No rotation
        translate=0.1,   # Translation fraction
        scale=0.5,       # Scaling factor
        shear=0.0,       # No shear
        perspective=0.0, # No perspective
        flipud=0.0,      # No vertical flip
        fliplr=0.5,      # Horizontal flip probability
        # Optimizer settings
        optimizer='SGD'
    )
    
    return model, results

def convert_yolo_to_coco_predictions(model, val_image_ids, image_info, output_file, conf_threshold=0.001):
    """Run inference and convert YOLO predictions to COCO format"""
    predictions = []
    
    for image_id in val_image_ids:
        img_info = image_info[image_id]
        img_filename = img_info['file_name']
        img_path = f"yolo_dataset/val/images/{img_filename}"
        
        if not pathlib.Path(img_path).exists():
            continue
            
        # Run inference
        results = model(img_path, conf=conf_threshold, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    # Get box coordinates in xyxy format
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # Convert xyxy to xywh (COCO format)
                    x1, y1, x2, y2 = xyxy
                    x = float(x1)
                    y = float(y1)
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    
                    prediction = {
                        "image_id": image_id,
                        "category_id": cls,  # Multi-class: preserve predicted class
                        "bbox": [x, y, w, h],
                        "score": float(conf)
                    }
                    predictions.append(prediction)
    
    # Save predictions
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    return predictions

def evaluate_detection_map(gt_ann_file, pred_file, val_image_ids):
    """Evaluate detection mAP@0.5 (category agnostic)"""
    
    # Load ground truth
    with open(gt_ann_file, 'r') as f:
        gt_data = json.load(f)
    
    # Filter ground truth to validation images only
    val_images = [img for img in gt_data['images'] if img['id'] in val_image_ids]
    val_annotations = [ann for ann in gt_data['annotations'] if ann['image_id'] in val_image_ids]
    
    # Create validation ground truth file for detection (single class)
    val_gt_data = {
        'images': val_images,
        'annotations': val_annotations.copy(),
        'categories': [{'id': 0, 'name': 'product', 'supercategory': 'product'}]  # Single class for detection
    }
    
    # Convert all category_ids to 0 for detection evaluation
    for ann in val_gt_data['annotations']:
        ann['category_id'] = 0
    
    # Load predictions and convert to single class for detection eval
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    # Convert predictions to single class for detection evaluation
    det_predictions = []
    for pred in predictions:
        det_pred = pred.copy()
        det_pred['category_id'] = 0  # Map all predictions to class 0
        det_predictions.append(det_pred)
    
    # Save temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(val_gt_data, f)
        val_gt_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(det_predictions, f)
        det_pred_file = f.name
    
    try:
        # Load COCO ground truth and predictions
        coco_gt = COCO(val_gt_file)
        coco_dt = coco_gt.loadRes(det_pred_file)
        
        # Run evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.catIds = [0]  # Single class
        coco_eval.params.iouThrs = [0.5]  # mAP@0.5 only
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract mAP@0.5
        map_50 = coco_eval.stats[1]  # mAP@0.5
        
        return map_50
        
    finally:
        # Clean up temporary files
        pathlib.Path(val_gt_file).unlink(missing_ok=True)
        pathlib.Path(det_pred_file).unlink(missing_ok=True)

def evaluate_classification_map(gt_ann_file, pred_file, val_image_ids, categories):
    """Evaluate classification mAP@0.5 (category aware)"""
    
    # Load ground truth
    with open(gt_ann_file, 'r') as f:
        gt_data = json.load(f)
    
    # Filter ground truth to validation images only
    val_images = [img for img in gt_data['images'] if img['id'] in val_image_ids]
    val_annotations = [ann for ann in gt_data['annotations'] if ann['image_id'] in val_image_ids]
    
    # Create validation ground truth file for classification (all categories)
    val_gt_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': [{'id': cat_id, 'name': cat_name, 'supercategory': 'product'} 
                      for cat_id, cat_name in categories.items()]
    }
    
    # Load predictions (keep original category_ids)
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    # Save temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(val_gt_data, f)
        val_gt_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(predictions, f)
        cls_pred_file = f.name
    
    try:
        # Load COCO ground truth and predictions
        coco_gt = COCO(val_gt_file)
        coco_dt = coco_gt.loadRes(cls_pred_file)
        
        # Run evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.iouThrs = [0.5]  # mAP@0.5 only
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract mAP@0.5
        map_50 = coco_eval.stats[1]  # mAP@0.5
        
        return map_50
        
    finally:
        # Clean up temporary files
        pathlib.Path(val_gt_file).unlink(missing_ok=True)
        pathlib.Path(cls_pred_file).unlink(missing_ok=True)

def main():
    print("=== YOLOv8m Multi-Class Detection Baseline ===")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Paths
    coco_ann_file = 'data/train/annotations.json'
    images_dir = 'data/train/images'
    yolo_dataset_dir = 'yolo_dataset'
    
    # Step 1: Convert COCO to YOLO format with all categories (0-356)
    print("\n1. Converting COCO annotations to YOLO format (multi-class)...")
    train_image_ids, val_image_ids, categories = convert_coco_to_yolo_multiclass(
        coco_ann_file, images_dir, yolo_dataset_dir, train_ratio=0.8
    )
    
    # Step 2: Train YOLOv8m model
    print("\n2. Training YOLOv8m model...")
    dataset_yaml_path = f"{yolo_dataset_dir}/dataset.yaml"
    model, results = train_yolo_model(dataset_yaml_path, epochs=50, imgsz=1280)
    
    # Step 3: Load best model for inference
    print("\n3. Loading best trained model...")
    best_model_path = 'runs/detect/yolov8m_multiclass/weights/best.pt'
    model = YOLO(best_model_path)
    
    # Step 4: Generate predictions on validation set
    print("\n4. Generating predictions on validation set...")
    with open(coco_ann_file, 'r') as f:
        coco_data = json.load(f)
    image_info = {img['id']: img for img in coco_data['images']}
    
    predictions_file = 'predictions.json'
    predictions = convert_yolo_to_coco_predictions(
        model, val_image_ids, image_info, predictions_file
    )
    
    print(f"Generated {len(predictions)} predictions")
    
    # Step 5: Evaluate detection mAP@0.5 (category agnostic)
    print("\n5. Evaluating detection mAP@0.5 (category agnostic)...")
    detection_map_50 = evaluate_detection_map(
        coco_ann_file, predictions_file, val_image_ids
    )
    
    # Step 6: Evaluate classification mAP@0.5 (category aware)
    print("\n6. Evaluating classification mAP@0.5 (category aware)...")
    classification_map_50 = evaluate_classification_map(
        coco_ann_file, predictions_file, val_image_ids, categories
    )
    
    # Calculate final score
    final_score = 0.7 * detection_map_50 + 0.3 * classification_map_50
    
    print(f"\nDetection mAP@0.5: {detection_map_50:.4f}")
    print(f"Classification mAP@0.5: {classification_map_50:.4f}")
    print(f"Final Score: {final_score:.4f}")
    
    # Print metrics in required format
    print(f"METRIC:detection_map_50={detection_map_50:.4f}")
    print(f"METRIC:classification_map_50={classification_map_50:.4f}")
    print(f"METRIC:final_score={final_score:.4f}")
    print(f"METRIC:val_predictions={len(predictions)}")
    print(f"METRIC:train_images={len(train_image_ids)}")
    print(f"METRIC:val_images={len(val_image_ids)}")
    print(f"METRIC:num_classes={len(categories)}")
    print(f"METRIC:model_size=yolov8m")
    print(f"METRIC:epochs=50")
    print(f"METRIC:batch_size=6")
    print(f"METRIC:imgsz=1280")
    
    print("\n=== YOLOv8m Multi-Class Detection Baseline Complete ===")
    print(f"Detection mAP@0.5: {detection_map_50:.4f}")
    print(f"Classification mAP@0.5: {classification_map_50:.4f}")
    print(f"Final Score (70% detection + 30% classification): {final_score:.4f}")
    print(f"Predictions saved to: {predictions_file}")
    print(f"Model saved to: {best_model_path}")

if __name__ == "__main__":
    main()