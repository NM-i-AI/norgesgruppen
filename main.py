import json
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import random

def convert_coco_to_yolo_single_class():
    """Convert COCO annotations to YOLO format with single class (product)"""
    print("Converting COCO annotations to YOLO format...")
    
    # Load COCO annotations
    with open('data/train/annotations.json', 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    yolo_dir = Path('data/yolo')
    yolo_dir.mkdir(exist_ok=True)
    (yolo_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (yolo_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (yolo_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (yolo_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    # Create image_id to filename mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Split images into train/val (90/10)
    image_ids = list(image_info.keys())
    random.seed(42)  # For reproducibility
    random.shuffle(image_ids)
    
    split_idx = int(0.9 * len(image_ids))
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]
    
    print(f"Train images: {len(train_ids)}, Val images: {len(val_ids)}")
    
    def process_split(image_ids, split_name):
        """Process train or val split"""
        for image_id in image_ids:
            img_info = image_info[image_id]
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Copy image (create symlink to save space)
            src_path = Path('data/train/images') / img_filename
            dst_path = yolo_dir / 'images' / split_name / img_filename
            
            if src_path.exists():
                # Create symlink instead of copying to save space
                if not dst_path.exists():
                    dst_path.symlink_to(src_path.resolve())
            
            # Convert annotations to YOLO format
            yolo_annotations = []
            for ann in annotations_by_image[image_id]:
                # COCO bbox format: [x, y, width, height] (top-left corner)
                x, y, w, h = ann['bbox']
                
                # Convert to YOLO format: [class_id, x_center, y_center, width, height] (normalized)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                
                # Single class (0 for all products)
                class_id = 0
                
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
            
            # Write YOLO label file
            label_filename = img_filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
            label_path = yolo_dir / 'labels' / split_name / label_filename
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
    
    process_split(train_ids, 'train')
    process_split(val_ids, 'val')
    
    # Create dataset.yaml
    dataset_yaml = {
        'path': str(yolo_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # number of classes
        'names': ['product']  # class names
    }
    
    with open(yolo_dir / 'dataset.yaml', 'w') as f:
        import yaml
        yaml.dump(dataset_yaml, f)
    
    print(f"YOLO dataset created at {yolo_dir}")
    return yolo_dir / 'dataset.yaml'

def train_yolo_baseline(dataset_yaml_path):
    """Train YOLOv8m baseline model"""
    print("Training YOLOv8m baseline model...")
    
    # Initialize YOLOv8m model
    model = YOLO('yolov8m.pt')  # Load pretrained model
    
    # Training parameters
    results = model.train(
        data=str(dataset_yaml_path),
        epochs=30,
        imgsz=1280,
        batch=8,  # Adjust based on GPU memory
        device=0 if torch.cuda.is_available() else 'cpu',
        project='runs/detect',
        name='baseline',
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
        patience=10,
        # Detection-specific parameters
        max_det=300,  # High max detections for dense shelves
        conf=0.001,   # Low confidence threshold for training
        iou=0.7,      # NMS IoU threshold
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,  # No rotation for shelf images
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,   # No vertical flip for shelf images
        fliplr=0.5,   # Horizontal flip OK
        mosaic=1.0,
        mixup=0.0
    )
    
    return model, results

def evaluate_model(model, dataset_yaml_path):
    """Evaluate model on validation set"""
    print("Evaluating model on validation set...")
    
    # Run validation
    results = model.val(
        data=str(dataset_yaml_path),
        imgsz=1280,
        batch=8,
        conf=0.001,
        iou=0.7,
        max_det=300,
        save_json=True,
        save_hybrid=False,
        plots=True,
        verbose=True
    )
    
    return results

def create_submission_format(model, test_images_dir=None):
    """Create submission format predictions (for validation set as proxy)"""
    print("Creating submission format predictions...")
    
    # For now, just demonstrate the format with validation predictions
    # In a real scenario, this would run on test images
    
    # Load validation images info
    with open('data/train/annotations.json', 'r') as f:
        coco_data = json.load(f)
    
    # Get validation image IDs (last 10% as we did in split)
    image_ids = [img['id'] for img in coco_data['images']]
    random.seed(42)
    random.shuffle(image_ids)
    split_idx = int(0.9 * len(image_ids))
    val_ids = image_ids[split_idx:]
    
    submission = []
    
    # Run inference on a few validation images as example
    for i, image_id in enumerate(val_ids[:5]):  # Just first 5 for demo
        img_info = next(img for img in coco_data['images'] if img['id'] == image_id)
        img_path = Path('data/train/images') / img_info['file_name']
        
        if img_path.exists():
            # Run inference
            results = model.predict(
                source=str(img_path),
                imgsz=1280,
                conf=0.25,  # Higher confidence for final predictions
                iou=0.7,
                max_det=300,
                verbose=False
            )
            
            # Convert to submission format
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    scores = result.boxes.conf.cpu().numpy()
                    
                    for box, score in zip(boxes, scores):
                        x1, y1, x2, y2 = box
                        # Convert to COCO format [x, y, width, height]
                        x, y, w, h = x1, y1, x2 - x1, y2 - y1
                        
                        submission.append({
                            "image_id": image_id,
                            "category_id": 0,  # All predictions as category 0
                            "bbox": [float(x), float(y), float(w), float(h)],
                            "score": float(score)
                        })
    
    print(f"Generated {len(submission)} predictions for {len(val_ids[:5])} validation images")
    return submission

def main():
    """Main experiment function"""
    print("=== YOLOv8m Baseline Single-Class Detection Experiment ===")
    
    try:
        # Step 1: Convert COCO to YOLO format
        dataset_yaml_path = convert_coco_to_yolo_single_class()
        
        # Step 2: Train YOLOv8m model
        model, train_results = train_yolo_baseline(dataset_yaml_path)
        
        # Step 3: Evaluate model
        val_results = evaluate_model(model, dataset_yaml_path)
        
        # Step 4: Extract metrics
        # YOLOv8 validation results contain mAP metrics
        if hasattr(val_results, 'box'):
            map50 = val_results.box.map50  # mAP@0.5
            map = val_results.box.map      # mAP@0.5:0.95
            precision = val_results.box.mp  # mean precision
            recall = val_results.box.mr     # mean recall
        else:
            # Fallback if structure is different
            map50 = 0.0
            map = 0.0
            precision = 0.0
            recall = 0.0
        
        # Calculate final score (detection only, so 70% of map50)
        final_score = 0.7 * map50
        
        # Step 5: Create example submission format
        submission = create_submission_format(model)
        
        # Print metrics
        print(f"\n=== Results ===")
        print(f"METRIC:detection_map50={map50:.4f}")
        print(f"METRIC:detection_map={map:.4f}")
        print(f"METRIC:precision={precision:.4f}")
        print(f"METRIC:recall={recall:.4f}")
        print(f"METRIC:final_score={final_score:.4f}")
        print(f"METRIC:submission_predictions={len(submission)}")
        
        # Success criteria check
        if map50 > 0.4:
            print(f"\n✅ SUCCESS: Detection mAP@0.5 ({map50:.4f}) > 0.4 threshold")
        else:
            print(f"\n❌ BELOW THRESHOLD: Detection mAP@0.5 ({map50:.4f}) <= 0.4")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Print error metrics
        print(f"METRIC:detection_map50=0.0")
        print(f"METRIC:final_score=0.0")
        print(f"METRIC:error=1")

if __name__ == "__main__":
    main()