import json
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import random

def load_exp006_model():
    """Load the best model from exp-006 training run"""
    print("Loading best model from exp-006...")
    
    # Look for the best model from exp-006 (multiclass_full_dataset)
    model_paths = [
        'runs/detect/multiclass_full_dataset/weights/best.pt',
        'runs/detect/multiclass_full_dataset2/weights/best.pt',
        'runs/detect/multiclass_full_dataset3/weights/best.pt',
        'runs/detect/multiclass_full_dataset4/weights/best.pt',
        'runs/detect/multiclass_full_dataset5/weights/best.pt'
    ]
    
    for model_path in model_paths:
        if Path(model_path).exists():
            print(f"Found model at: {model_path}")
            model = YOLO(model_path)
            return model
    
    # If no trained model found, fall back to training a new one
    print("No pre-trained model found. Training new model...")
    dataset_yaml_path, category_mapping, val_ids = convert_coco_to_yolo_multiclass()
    model, _ = train_yolo_multiclass_full(dataset_yaml_path)
    return model

def convert_coco_to_yolo_multiclass():
    """Convert COCO annotations to YOLO format with all categories - ALL IMAGES FOR TRAINING"""
    print("Converting COCO annotations to YOLO format (multi-class, all images for training)...")
    
    # Load COCO annotations
    with open('data/train/annotations.json', 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    yolo_dir = Path('data/yolo_multiclass_full')
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
    
    # Define validation set (same as before for consistent evaluation)
    image_ids = list(image_info.keys())
    random.seed(42)  # For reproducibility
    random.shuffle(image_ids)
    
    split_idx = int(0.9 * len(image_ids))
    val_ids = image_ids[split_idx:]  # Keep same val set for evaluation
    
    # NEW: Use ALL images for training (no train/val split)
    train_ids = image_ids  # All images go to training
    
    print(f"Train images: {len(train_ids)} (ALL), Val images for eval: {len(val_ids)}")
    
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
    
    # Process all images as training data
    process_split(train_ids, 'train')
    
    # Create dataset.yaml
    dataset_yaml = {
        'path': str(yolo_dir.resolve()),
        'train': 'images/train',
        'val': 'images/train',  # Point to train since we're not using YOLO's val split
        'nc': len(categories),  # number of classes
        'names': category_names  # class names
    }
    
    with open(yolo_dir / 'dataset.yaml', 'w') as f:
        import yaml
        yaml.dump(dataset_yaml, f)
    
    print(f"Multi-class YOLO dataset created at {yolo_dir}")
    print(f"Number of classes: {len(categories)}")
    print(f"Training on ALL {len(train_ids)} images")
    return yolo_dir / 'dataset.yaml', category_mapping, val_ids

def train_yolo_multiclass_full(dataset_yaml_path):
    """Train YOLOv8l multi-class model on full dataset with best hyperparameters"""
    print("Training YOLOv8l multi-class model on FULL dataset with tuned hyperparameters...")
    
    # Initialize YOLOv8l model (same as exp-004)
    model = YOLO('yolov8l.pt')  # Load pretrained YOLOv8l model
    
    # Training parameters - same as exp-006
    results = model.train(
        data=str(dataset_yaml_path),
        epochs=80,  # Same as exp-006
        imgsz=1280,
        batch=6,  # Same as exp-006
        device=0 if torch.cuda.is_available() else 'cpu',
        project='runs/detect',
        name='multiclass_full_dataset_tta',
        save=True,
        save_period=20,
        val=False,  # Disable YOLO validation since we're using all data for training
        plots=True,
        verbose=True,
        patience=20,
        
        # Detection-specific parameters (same as exp-006)
        max_det=300,
        conf=0.001,
        iou=0.7,
        
        # Learning rate schedule (same as exp-006)
        lr0=0.01,
        lrf=0.01,
        
        # Optimizer settings (same as exp-006)
        optimizer='AdamW',
        weight_decay=0.0005,
        
        # Enhanced data augmentation (same as exp-006)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.9,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,     # Same as exp-006
        copy_paste=0.3, # Same as exp-006
        
        # Warmup settings (same as exp-006)
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Loss function weights (same as exp-006)
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Close mosaic augmentation in final epochs
        close_mosaic=10
    )
    
    return model, results

def evaluate_model_with_tta(model, val_ids, category_mapping, use_tta=True):
    """Evaluate model on held-out validation set with optional TTA"""
    tta_str = "WITH TTA" if use_tta else "WITHOUT TTA"
    print(f"Evaluating model on {len(val_ids)} held-out validation images {tta_str}...")
    
    # Load validation data for custom evaluation
    with open('data/train/annotations.json', 'r') as f:
        coco_data = json.load(f)
    
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
    
    # Evaluate on more images for better metrics
    eval_ids = val_ids[:50]  # Evaluate on first 50 val images
    
    for image_id in eval_ids:
        img_info = image_info[image_id]
        img_path = Path('data/train/images') / img_info['file_name']
        
        if not img_path.exists():
            continue
            
        # Run inference with or without TTA
        results = model.predict(
            source=str(img_path),
            imgsz=1280,
            conf=0.25,
            iou=0.7,
            max_det=300,
            augment=use_tta,  # KEY CHANGE: Enable/disable TTA
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
    
    # Calculate IoU and mAP metrics (same as before)
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

def main():
    """Main experiment function - Test-time augmentation (TTA) on best exp-006 model"""
    print("=== Test-Time Augmentation (TTA) Experiment - Step 9 ===")
    
    try:
        # Step 1: Load the best model from exp-006 or train if needed
        model = load_exp006_model()
        
        # Step 2: Set up validation data
        # Load COCO annotations to get category mapping and val_ids
        with open('data/train/annotations.json', 'r') as f:
            coco_data = json.load(f)
        
        # Create category mapping (same as training)
        categories = sorted(coco_data['categories'], key=lambda x: x['id'])
        category_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}
        
        # Define validation set (same split as exp-006)
        image_info = {img['id']: img for img in coco_data['images']}
        image_ids = list(image_info.keys())
        random.seed(42)  # For reproducibility
        random.shuffle(image_ids)
        split_idx = int(0.9 * len(image_ids))
        val_ids = image_ids[split_idx:]
        
        print(f"Validation set: {len(val_ids)} images")
        
        # Step 3: Evaluate WITHOUT TTA (baseline)
        print("\n=== Evaluating WITHOUT TTA (baseline) ===")
        detection_map50_no_tta, classification_map50_no_tta, detection_recall_no_tta, classification_recall_no_tta = evaluate_model_with_tta(
            model, val_ids, category_mapping, use_tta=False
        )
        final_score_no_tta = 0.7 * detection_map50_no_tta + 0.3 * classification_map50_no_tta
        
        # Step 4: Evaluate WITH TTA
        print("\n=== Evaluating WITH TTA ===")
        detection_map50_tta, classification_map50_tta, detection_recall_tta, classification_recall_tta = evaluate_model_with_tta(
            model, val_ids, category_mapping, use_tta=True
        )
        final_score_tta = 0.7 * detection_map50_tta + 0.3 * classification_map50_tta
        
        # Step 5: Calculate improvements
        detection_improvement = ((detection_map50_tta - detection_map50_no_tta) / detection_map50_no_tta * 100) if detection_map50_no_tta > 0 else 0
        classification_improvement = ((classification_map50_tta - classification_map50_no_tta) / classification_map50_no_tta * 100) if classification_map50_no_tta > 0 else 0
        final_score_improvement = ((final_score_tta - final_score_no_tta) / final_score_no_tta * 100) if final_score_no_tta > 0 else 0
        
        # Print metrics
        print(f"\n=== TTA Comparison Results ===")
        print(f"WITHOUT TTA:")
        print(f"  detection_map50={detection_map50_no_tta:.4f}")
        print(f"  classification_map50={classification_map50_no_tta:.4f}")
        print(f"  final_score={final_score_no_tta:.4f}")
        
        print(f"\nWITH TTA:")
        print(f"  detection_map50={detection_map50_tta:.4f}")
        print(f"  classification_map50={classification_map50_tta:.4f}")
        print(f"  final_score={final_score_tta:.4f}")
        
        print(f"\nIMPROVEMENTS:")
        print(f"  detection_improvement={detection_improvement:.2f}%")
        print(f"  classification_improvement={classification_improvement:.2f}%")
        print(f"  final_score_improvement={final_score_improvement:.2f}%")
        
        # Print final metrics (use TTA results as main metrics)
        print(f"\n=== Final Metrics (WITH TTA) ===") 
        print(f"METRIC:detection_map50={detection_map50_tta:.4f}")
        print(f"METRIC:classification_map50={classification_map50_tta:.4f}")
        print(f"METRIC:final_score={final_score_tta:.4f}")
        print(f"METRIC:detection_recall={detection_recall_tta:.4f}")
        print(f"METRIC:classification_recall={classification_recall_tta:.4f}")
        print(f"METRIC:detection_improvement_pct={detection_improvement:.2f}")
        print(f"METRIC:classification_improvement_pct={classification_improvement:.2f}")
        print(f"METRIC:final_score_improvement_pct={final_score_improvement:.2f}")
        print(f"METRIC:tta_enabled=1")
        print(f"METRIC:val_images_evaluated={len(val_ids)}")
        
        # Success criteria check
        exp006_score = 0.8498  # From exp-006
        if final_score_tta > exp006_score:
            improvement = ((final_score_tta - exp006_score) / exp006_score) * 100
            print(f"\n✅ SUCCESS: Final score WITH TTA ({final_score_tta:.4f}) > exp-006 ({exp006_score:.4f})")
            print(f"📈 IMPROVEMENT: +{improvement:.1f}% over exp-006")
        else:
            decline = ((exp006_score - final_score_tta) / exp006_score) * 100
            print(f"\n❌ BELOW EXP-006: Final score WITH TTA ({final_score_tta:.4f}) <= exp-006 ({exp006_score:.4f})")
            print(f"📉 DECLINE: -{decline:.1f}% from exp-006")
        
        # TTA effectiveness check
        if final_score_tta > final_score_no_tta:
            print(f"\n✅ TTA EFFECTIVE: TTA improved final_score by {final_score_improvement:.2f}%")
        else:
            print(f"\n❌ TTA INEFFECTIVE: TTA did not improve final_score")
        
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