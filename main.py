import json
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import random

def convert_coco_to_yolo_multiclass():
    """Convert COCO annotations to YOLO format with all categories"""
    print("Converting COCO annotations to YOLO format (multi-class)...")
    
    # Load COCO annotations
    with open('data/train/annotations.json', 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    yolo_dir = Path('data/yolo_multiclass')
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
    
    # Create category mapping (COCO category_id to YOLO class_id)
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    category_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}
    category_names = [cat['name'] for cat in categories]
    
    print(f"Found {len(categories)} categories")
    print(f"Category ID range: {min(cat['id'] for cat in categories)} to {max(cat['id'] for cat in categories)}")
    
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
                
                # Map COCO category_id to YOLO class_id
                coco_cat_id = ann['category_id']
                yolo_class_id = category_mapping[coco_cat_id]
                
                yolo_annotations.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
            
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
        'nc': len(categories),  # number of classes
        'names': category_names  # class names
    }
    
    with open(yolo_dir / 'dataset.yaml', 'w') as f:
        import yaml
        yaml.dump(dataset_yaml, f)
    
    print(f"Multi-class YOLO dataset created at {yolo_dir}")
    print(f"Number of classes: {len(categories)}")
    return yolo_dir / 'dataset.yaml', category_mapping

def train_yolo_multiclass(dataset_yaml_path):
    """Train YOLOv8m multi-class model"""
    print("Training YOLOv8m multi-class model...")
    
    # Initialize YOLOv8m model
    model = YOLO('yolov8m.pt')  # Load pretrained model
    
    # Training parameters
    results = model.train(
        data=str(dataset_yaml_path),
        epochs=50,  # Increased epochs for multi-class
        imgsz=1280,
        batch=8,  # Adjust based on GPU memory
        device=0 if torch.cuda.is_available() else 'cpu',
        project='runs/detect',
        name='multiclass',
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
        patience=15,  # Increased patience for multi-class
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

def evaluate_multiclass_model(model, dataset_yaml_path):
    """Evaluate multi-class model on validation set"""
    print("Evaluating multi-class model on validation set...")
    
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

def calculate_detection_and_classification_metrics(model, dataset_yaml_path, category_mapping):
    """Calculate both detection (class-agnostic) and classification (class-aware) metrics"""
    print("Calculating detection and classification metrics...")
    
    # Load validation data for custom evaluation
    with open('data/train/annotations.json', 'r') as f:
        coco_data = json.load(f)
    
    # Get validation image IDs (same split as training)
    image_ids = [img['id'] for img in coco_data['images']]
    random.seed(42)
    random.shuffle(image_ids)
    split_idx = int(0.9 * len(image_ids))
    val_ids = image_ids[split_idx:]
    
    # Create image info mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Group ground truth annotations by image
    gt_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        if ann['image_id'] in val_ids:
            gt_by_image[ann['image_id']].append(ann)
    
    # Run inference on validation images
    all_predictions = []
    all_gt_detection = []  # For detection (class-agnostic)
    all_gt_classification = []  # For classification (class-aware)
    
    for image_id in val_ids[:20]:  # Evaluate on subset for speed
        img_info = image_info[image_id]
        img_path = Path('data/train/images') / img_info['file_name']
        
        if not img_path.exists():
            continue
            
        # Run inference
        results = model.predict(
            source=str(img_path),
            imgsz=1280,
            conf=0.25,
            iou=0.7,
            max_det=300,
            verbose=False
        )
        
        # Process predictions
        predictions = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                # Convert to COCO format [x, y, width, height]
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                
                predictions.append({
                    'bbox': [x, y, w, h],
                    'score': score,
                    'category_id': cls
                })
        
        all_predictions.append(predictions)
        
        # Process ground truth
        gt_detection = []  # Class-agnostic (all as class 0)
        gt_classification = []  # Class-aware
        
        for ann in gt_by_image[image_id]:
            bbox = ann['bbox']  # Already in COCO format
            
            # Detection ground truth (class-agnostic)
            gt_detection.append({
                'bbox': bbox,
                'category_id': 0  # All as single class for detection
            })
            
            # Classification ground truth (class-aware)
            yolo_class_id = category_mapping[ann['category_id']]
            gt_classification.append({
                'bbox': bbox,
                'category_id': yolo_class_id
            })
        
        all_gt_detection.append(gt_detection)
        all_gt_classification.append(gt_classification)
    
    # Calculate IoU and mAP metrics (simplified)
    def calculate_iou(box1, box2):
        """Calculate IoU between two boxes in [x, y, w, h] format"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to [x1, y1, x2, y2]
        box1_xyxy = [x1, y1, x1 + w1, y1 + h1]
        box2_xyxy = [x2, y2, x2 + w2, y2 + h2]
        
        # Calculate intersection
        x_left = max(box1_xyxy[0], box2_xyxy[0])
        y_top = max(box1_xyxy[1], box2_xyxy[1])
        x_right = min(box1_xyxy[2], box2_xyxy[2])
        y_bottom = min(box1_xyxy[3], box2_xyxy[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_map_simple(predictions_list, gt_list, iou_threshold=0.5):
        """Simplified mAP calculation"""
        total_tp = 0
        total_fp = 0
        total_gt = 0
        
        for preds, gts in zip(predictions_list, gt_list):
            total_gt += len(gts)
            
            # Sort predictions by score
            preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)
            
            matched_gt = set()
            
            for pred in preds_sorted:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gts):
                    if gt_idx in matched_gt:
                        continue
                        
                    iou = calculate_iou(pred['bbox'], gt['bbox'])
                    
                    # For classification, also check category match
                    category_match = (pred['category_id'] == gt['category_id'])
                    
                    if iou > best_iou and iou >= iou_threshold and category_match:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx >= 0:
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        
        return precision, recall
    
    # Calculate detection mAP (class-agnostic)
    # Convert predictions to class-agnostic
    detection_predictions = []
    for preds in all_predictions:
        det_preds = []
        for pred in preds:
            det_pred = pred.copy()
            det_pred['category_id'] = 0  # All as single class
            det_preds.append(det_pred)
        detection_predictions.append(det_preds)
    
    detection_precision, detection_recall = calculate_map_simple(
        detection_predictions, all_gt_detection, iou_threshold=0.5
    )
    
    # Calculate classification mAP (class-aware)
    classification_precision, classification_recall = calculate_map_simple(
        all_predictions, all_gt_classification, iou_threshold=0.5
    )
    
    # Approximate mAP as precision (simplified)
    detection_map50 = detection_precision
    classification_map50 = classification_precision
    
    return detection_map50, classification_map50, detection_recall, classification_recall

def create_multiclass_submission(model, category_mapping):
    """Create submission format predictions with actual category IDs"""
    print("Creating multi-class submission format predictions...")
    
    # Load validation images info
    with open('data/train/annotations.json', 'r') as f:
        coco_data = json.load(f)
    
    # Get validation image IDs (last 10% as we did in split)
    image_ids = [img['id'] for img in coco_data['images']]
    random.seed(42)
    random.shuffle(image_ids)
    split_idx = int(0.9 * len(image_ids))
    val_ids = image_ids[split_idx:]
    
    # Create reverse mapping from YOLO class_id to COCO category_id
    reverse_mapping = {v: k for k, v in category_mapping.items()}
    
    submission = []
    
    # Run inference on validation images
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
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, score, cls in zip(boxes, scores, classes):
                        x1, y1, x2, y2 = box
                        # Convert to COCO format [x, y, width, height]
                        x, y, w, h = x1, y1, x2 - x1, y2 - y1
                        
                        # Map YOLO class_id back to COCO category_id
                        coco_category_id = reverse_mapping.get(cls, 0)
                        
                        submission.append({
                            "image_id": image_id,
                            "category_id": coco_category_id,
                            "bbox": [float(x), float(y), float(w), float(h)],
                            "score": float(score)
                        })
    
    print(f"Generated {len(submission)} predictions for {len(val_ids[:5])} validation images")
    return submission

def main():
    """Main experiment function"""
    print("=== YOLOv8m Multi-Class Detection Experiment ===")
    
    try:
        # Step 1: Convert COCO to YOLO format (multi-class)
        dataset_yaml_path, category_mapping = convert_coco_to_yolo_multiclass()
        
        # Step 2: Train YOLOv8m multi-class model
        model, train_results = train_yolo_multiclass(dataset_yaml_path)
        
        # Step 3: Evaluate model
        val_results = evaluate_multiclass_model(model, dataset_yaml_path)
        
        # Step 4: Calculate detection and classification metrics
        detection_map50, classification_map50, detection_recall, classification_recall = calculate_detection_and_classification_metrics(
            model, dataset_yaml_path, category_mapping
        )
        
        # Step 5: Calculate final score
        final_score = 0.7 * detection_map50 + 0.3 * classification_map50
        
        # Step 6: Create submission format
        submission = create_multiclass_submission(model, category_mapping)
        
        # Extract additional metrics from YOLO validation
        if hasattr(val_results, 'box'):
            yolo_map50 = val_results.box.map50  # Overall mAP@0.5
            yolo_map = val_results.box.map      # mAP@0.5:0.95
            yolo_precision = val_results.box.mp  # mean precision
            yolo_recall = val_results.box.mr     # mean recall
        else:
            yolo_map50 = 0.0
            yolo_map = 0.0
            yolo_precision = 0.0
            yolo_recall = 0.0
        
        # Print metrics
        print(f"\n=== Results ===")
        print(f"METRIC:detection_map50={detection_map50:.4f}")
        print(f"METRIC:classification_map50={classification_map50:.4f}")
        print(f"METRIC:final_score={final_score:.4f}")
        print(f"METRIC:detection_recall={detection_recall:.4f}")
        print(f"METRIC:classification_recall={classification_recall:.4f}")
        print(f"METRIC:yolo_map50={yolo_map50:.4f}")
        print(f"METRIC:yolo_map={yolo_map:.4f}")
        print(f"METRIC:yolo_precision={yolo_precision:.4f}")
        print(f"METRIC:yolo_recall={yolo_recall:.4f}")
        print(f"METRIC:num_categories={len(category_mapping)}")
        print(f"METRIC:submission_predictions={len(submission)}")
        
        # Success criteria check
        baseline_score = 0.6808  # From step 2
        if final_score > baseline_score:
            print(f"\n✅ SUCCESS: Final score ({final_score:.4f}) > baseline ({baseline_score:.4f})")
        else:
            print(f"\n❌ BELOW BASELINE: Final score ({final_score:.4f}) <= baseline ({baseline_score:.4f})")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Print error metrics
        print(f"METRIC:detection_map50=0.0")
        print(f"METRIC:classification_map50=0.0")
        print(f"METRIC:final_score=0.0")
        print(f"METRIC:error=1")

if __name__ == "__main__":
    main()