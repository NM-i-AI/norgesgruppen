import json
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import random
from itertools import product

def load_best_model():
    """Load the best trained model from exp-004"""
    print("Loading best trained model from exp-004...")
    
    # Try to find the best model from previous training
    model_paths = [
        'runs/detect/multiclass_scaled/weights/best.pt',
        'runs/detect/multiclass_scaled/weights/last.pt',
        'best.pt',  # If saved in current directory
        'last.pt'
    ]
    
    for model_path in model_paths:
        if Path(model_path).exists():
            print(f"Found model at: {model_path}")
            return YOLO(model_path)
    
    # If no trained model found, train a quick one
    print("No trained model found, training a quick YOLOv8l model...")
    dataset_yaml_path, category_mapping = convert_coco_to_yolo_multiclass()
    model = YOLO('yolov8l.pt')
    
    # Quick training with minimal epochs
    model.train(
        data=str(dataset_yaml_path),
        epochs=10,  # Minimal for time constraints
        imgsz=1280,
        batch=8,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='runs/detect',
        name='quick_multiclass',
        save=True,
        verbose=False
    )
    
    return model

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
    
    # Split images into train/val (90/10)
    image_ids = list(image_info.keys())
    random.seed(42)  # For reproducibility
    random.shuffle(image_ids)
    
    split_idx = int(0.9 * len(image_ids))
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]
    
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
    
    return yolo_dir / 'dataset.yaml', category_mapping

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

def calculate_precision_recall_curve(predictions_list, gt_list, iou_threshold=0.5):
    """Calculate precision-recall curve for mAP calculation"""
    # Collect all predictions with scores
    all_predictions = []
    for img_idx, preds in enumerate(predictions_list):
        for pred in preds:
            all_predictions.append({
                'image_idx': img_idx,
                'score': pred['score'],
                'bbox': pred['bbox'],
                'category_id': pred['category_id']
            })
    
    # Sort by confidence score (descending)
    all_predictions.sort(key=lambda x: x['score'], reverse=True)
    
    # Count total ground truth boxes
    total_gt = sum(len(gts) for gts in gt_list)
    
    if total_gt == 0:
        return 0.0  # No ground truth
    
    # Calculate precision and recall at each threshold
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    
    # Track which ground truth boxes have been matched
    matched_gt = [set() for _ in range(len(gt_list))]
    
    for pred in all_predictions:
        img_idx = pred['image_idx']
        pred_bbox = pred['bbox']
        pred_category = pred['category_id']
        
        # Find best matching ground truth box
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(gt_list[img_idx]):
            if gt_idx in matched_gt[img_idx]:
                continue  # Already matched
            
            iou = calculate_iou(pred_bbox, gt['bbox'])
            
            # Check if category matches (for classification)
            category_match = (pred_category == gt['category_id'])
            
            if iou > best_iou and iou >= iou_threshold and category_match:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Update TP/FP
        if best_gt_idx >= 0:
            tp += 1
            matched_gt[img_idx].add(best_gt_idx)
        else:
            fp += 1
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_gt
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AP using 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        # Find precisions for recalls >= t
        valid_precisions = [p for p, r in zip(precisions, recalls) if r >= t]
        if valid_precisions:
            ap += max(valid_precisions) / 11
    
    return ap

def run_inference_with_thresholds(model, val_image_paths, conf_threshold, nms_iou):
    """Run inference with specific confidence and NMS thresholds"""
    all_predictions = []
    
    for img_path in val_image_paths:
        # Run inference
        results = model.predict(
            source=str(img_path),
            imgsz=1280,
            conf=conf_threshold,
            iou=nms_iou,
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
    
    return all_predictions

def threshold_optimization_sweep(model, category_mapping):
    """Run comprehensive threshold optimization sweep"""
    print("Running confidence and NMS threshold optimization sweep...")
    
    # Load validation data
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
    
    # Prepare validation image paths and ground truth
    val_image_paths = []
    all_gt_detection = []  # For detection (class-agnostic)
    all_gt_classification = []  # For classification (class-aware)
    
    for image_id in val_ids:
        img_info = image_info[image_id]
        img_path = Path('data/train/images') / img_info['file_name']
        
        if not img_path.exists():
            continue
            
        val_image_paths.append(img_path)
        
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
    
    print(f"Evaluating on {len(val_image_paths)} validation images")
    
    # Define threshold ranges to sweep
    conf_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    nms_ious = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    best_score = 0
    best_params = None
    best_metrics = None
    
    results = []
    
    print(f"Testing {len(conf_thresholds)} confidence × {len(nms_ious)} NMS = {len(conf_thresholds) * len(nms_ious)} combinations")
    
    for i, (conf_thresh, nms_iou) in enumerate(product(conf_thresholds, nms_ious)):
        print(f"\rProgress: {i+1}/{len(conf_thresholds) * len(nms_ious)} - conf={conf_thresh:.2f}, nms={nms_iou:.1f}", end="")
        
        # Run inference with current thresholds
        predictions = run_inference_with_thresholds(model, val_image_paths, conf_thresh, nms_iou)
        
        # Calculate detection mAP (class-agnostic)
        detection_predictions = []
        for preds in predictions:
            det_preds = []
            for pred in preds:
                det_pred = pred.copy()
                det_pred['category_id'] = 0  # All as single class
                det_preds.append(det_pred)
            detection_predictions.append(det_preds)
        
        detection_map50 = calculate_precision_recall_curve(
            detection_predictions, all_gt_detection, iou_threshold=0.5
        )
        
        # Calculate classification mAP (class-aware)
        classification_map50 = calculate_precision_recall_curve(
            predictions, all_gt_classification, iou_threshold=0.5
        )
        
        # Calculate final score
        final_score = 0.7 * detection_map50 + 0.3 * classification_map50
        
        # Store results
        result = {
            'conf_threshold': conf_thresh,
            'nms_iou': nms_iou,
            'detection_map50': detection_map50,
            'classification_map50': classification_map50,
            'final_score': final_score,
            'total_predictions': sum(len(preds) for preds in predictions)
        }
        results.append(result)
        
        # Track best result
        if final_score > best_score:
            best_score = final_score
            best_params = (conf_thresh, nms_iou)
            best_metrics = result
    
    print("\n")
    
    # Sort results by final score
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    return results, best_params, best_metrics

def main():
    """Main experiment function"""
    print("=== Confidence Threshold and NMS Optimization Sweep ===")
    
    try:
        # Step 1: Load best trained model
        model = load_best_model()
        
        # Step 2: Convert COCO to YOLO format to get category mapping
        dataset_yaml_path, category_mapping = convert_coco_to_yolo_multiclass()
        
        # Step 3: Run threshold optimization sweep
        results, best_params, best_metrics = threshold_optimization_sweep(model, category_mapping)
        
        # Step 4: Print detailed results
        print(f"\n=== Optimization Results ===")
        print(f"Best parameters: conf={best_params[0]:.3f}, nms_iou={best_params[1]:.1f}")
        print(f"Best final_score: {best_metrics['final_score']:.4f}")
        print(f"Best detection_map50: {best_metrics['detection_map50']:.4f}")
        print(f"Best classification_map50: {best_metrics['classification_map50']:.4f}")
        print(f"Total predictions: {best_metrics['total_predictions']}")
        
        print(f"\nTop 10 configurations:")
        for i, result in enumerate(results[:10]):
            print(f"  {i+1:2d}. conf={result['conf_threshold']:.3f}, nms={result['nms_iou']:.1f} → "
                  f"final={result['final_score']:.4f} (det={result['detection_map50']:.4f}, "
                  f"cls={result['classification_map50']:.4f}, preds={result['total_predictions']})")
        
        # Print metrics in required format
        print(f"\n=== Final Metrics ===")
        print(f"METRIC:best_conf_threshold={best_params[0]:.3f}")
        print(f"METRIC:best_nms_iou={best_params[1]:.1f}")
        print(f"METRIC:detection_map50={best_metrics['detection_map50']:.4f}")
        print(f"METRIC:classification_map50={best_metrics['classification_map50']:.4f}")
        print(f"METRIC:final_score={best_metrics['final_score']:.4f}")
        print(f"METRIC:total_predictions={best_metrics['total_predictions']}")
        print(f"METRIC:num_configurations_tested={len(results)}")
        print(f"METRIC:num_validation_images={len([r for r in results if r == results[0]])}")
        
        # Compare to baseline
        baseline_score = 0.7882  # From exp-004
        if best_metrics['final_score'] > baseline_score:
            improvement = ((best_metrics['final_score'] - baseline_score) / baseline_score) * 100
            print(f"METRIC:improvement_pct={improvement:.2f}")
            print(f"\n✅ SUCCESS: Optimized score ({best_metrics['final_score']:.4f}) > baseline ({baseline_score:.4f})")
        else:
            decline = ((baseline_score - best_metrics['final_score']) / baseline_score) * 100
            print(f"METRIC:decline_pct={decline:.2f}")
            print(f"\n❌ NO IMPROVEMENT: Optimized score ({best_metrics['final_score']:.4f}) <= baseline ({baseline_score:.4f})")
        
        # Success criteria check
        target_score = 0.79
        if best_metrics['final_score'] > target_score:
            print(f"✅ TARGET MET: Final score ({best_metrics['final_score']:.4f}) > target ({target_score:.4f})")
        else:
            print(f"❌ TARGET MISSED: Final score ({best_metrics['final_score']:.4f}) <= target ({target_score:.4f})")
        
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