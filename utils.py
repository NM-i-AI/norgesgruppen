"""Shared utilities for grocery product detection.

Key functions:
- evaluate_model: Compute val_score using pycocotools
- prepare_yolo_dataset: Create YOLO dataset YAML configuration
"""

import json
import yaml
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile
import numpy as np

def evaluate_model(predictions, val_image_ids, annotations_path):
    """
    Evaluate model predictions using the competition metric.
    
    Args:
        predictions: List of prediction dicts with keys:
                    ['image_id', 'category_id', 'bbox', 'score']
        val_image_ids: Set of image IDs to evaluate on
        annotations_path: Path to COCO annotations file
    
    Returns:
        tuple: (val_score, detection_mAP@0.5, classification_mAP@0.5)
               where val_score = 0.7 * detection_mAP + 0.3 * classification_mAP
    """
    
    # Load ground truth annotations
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Filter to validation images only
    val_images = [img for img in coco_data['images'] if img['id'] in val_image_ids]
    val_annotations = [ann for ann in coco_data['annotations'] 
                      if ann['image_id'] in val_image_ids]
    
    # Create filtered COCO data for validation
    val_coco_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': coco_data['categories']
    }
    
    # Write to temporary file for COCO API
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(val_coco_data, f)
        temp_gt_path = f.name
    
    try:
        # Initialize COCO ground truth
        coco_gt = COCO(temp_gt_path)
        
        # Filter predictions to validation images
        val_predictions = [pred for pred in predictions 
                          if pred['image_id'] in val_image_ids]
        
        if not val_predictions:
            print("Warning: No predictions for validation images")
            return 0.0, 0.0, 0.0
        
        # Load predictions into COCO format
        coco_dt = coco_gt.loadRes(val_predictions)
        
        # 1. DETECTION EVALUATION (category-agnostic)
        # Convert all predictions and ground truth to single class for detection eval
        detection_predictions = []
        for pred in val_predictions:
            det_pred = pred.copy()
            det_pred['category_id'] = 1  # Single class for detection
            detection_predictions.append(det_pred)
        
        # Create detection ground truth (all annotations become class 1)
        detection_gt_data = val_coco_data.copy()
        detection_gt_data['categories'] = [{'id': 1, 'name': 'product'}]
        detection_gt_data['annotations'] = []
        for ann in val_annotations:
            det_ann = ann.copy()
            det_ann['category_id'] = 1
            detection_gt_data['annotations'].append(det_ann)
        
        # Write detection ground truth
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(detection_gt_data, f)
            temp_det_gt_path = f.name
        
        try:
            # Detection evaluation
            coco_det_gt = COCO(temp_det_gt_path)
            coco_det_dt = coco_det_gt.loadRes(detection_predictions)
            
            coco_eval_det = COCOeval(coco_det_gt, coco_det_dt, 'bbox')
            coco_eval_det.params.iouThrs = np.array([0.5])  # Only IoU@0.5
            coco_eval_det.params.maxDets = [100, 300, 1000]  # Standard maxDets
            coco_eval_det.evaluate()
            coco_eval_det.accumulate()
            coco_eval_det.summarize()
            
            # Extract mAP@0.5 for detection (index 1 in stats)
            detection_map = coco_eval_det.stats[1]  # mAP@0.5
            
        finally:
            Path(temp_det_gt_path).unlink(missing_ok=True)
        
        # 2. CLASSIFICATION EVALUATION (category-specific)
        coco_eval_cls = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval_cls.params.iouThrs = np.array([0.5])  # Only IoU@0.5
        coco_eval_cls.params.maxDets = [100, 300, 1000]  # Standard maxDets
        coco_eval_cls.evaluate()
        coco_eval_cls.accumulate()
        coco_eval_cls.summarize()
        
        # Extract mAP@0.5 for classification (index 1 in stats)
        classification_map = coco_eval_cls.stats[1]  # mAP@0.5
        
        # 3. COMPUTE COMBINED SCORE
        val_score = 0.7 * detection_map + 0.3 * classification_map
        
        return val_score, detection_map, classification_map
        
    finally:
        # Clean up temporary files
        Path(temp_gt_path).unlink(missing_ok=True)

def prepare_yolo_dataset():
    """
    Prepare YOLO dataset YAML configuration file.
    
    Returns:
        str: Path to created YAML file
    """
    
    data_dir = Path("data")
    annotations_path = data_dir / "train" / "annotations.json"
    
    # Load COCO annotations
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Extract category information
    categories = coco_data['categories']
    num_classes = len(categories)
    
    # Create class names mapping (YOLO expects 0-indexed)
    class_names = {}
    for cat in categories:
        class_names[cat['id']] = cat['name']
    
    # Create YOLO dataset configuration
    yolo_config = {
        'path': str(data_dir.absolute()),
        'train': 'train/images',  # Relative to path
        'val': 'train/images',    # Will be updated when we create actual splits
        'nc': num_classes,
        'names': class_names
    }
    
    # Save YAML configuration
    yaml_path = Path("grocery_dataset.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yolo_config, f, default_flow_style=False)
    
    return str(yaml_path)

def create_train_val_split(annotations_path, val_ratio=0.1, seed=42):
    """
    Create 90/10 train/val split stratified by store section if possible.
    
    Args:
        annotations_path: Path to COCO annotations file
        val_ratio: Fraction of images for validation (default 0.1 = 10%)
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_image_ids, val_image_ids)
    """
    
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    
    # Try to extract store section from image filenames if available
    # This is a heuristic - adjust based on actual filename patterns
    section_images = {}
    for img in images:
        filename = img['file_name']
        # Try to identify section from filename (this may need adjustment)
        section = 'unknown'
        for section_name in ['Egg', 'Frokost', 'Knekkebrod', 'Varmedrikker']:
            if section_name.lower() in filename.lower():
                section = section_name
                break
        
        if section not in section_images:
            section_images[section] = []
        section_images[section].append(img['id'])
    
    # Perform stratified split
    train_image_ids = set()
    val_image_ids = set()
    
    for section, image_ids in section_images.items():
        random.shuffle(image_ids)
        val_count = max(1, int(len(image_ids) * val_ratio))
        
        val_image_ids.update(image_ids[:val_count])
        train_image_ids.update(image_ids[val_count:])
    
    return train_image_ids, val_image_ids

def save_coco_split(coco_data, image_ids, output_path):
    """
    Save a subset of COCO data containing only specified images.
    
    Args:
        coco_data: Full COCO dataset dict
        image_ids: Set of image IDs to include
        output_path: Path to save filtered COCO data
    """
    
    # Filter images and annotations
    filtered_images = [img for img in coco_data['images'] if img['id'] in image_ids]
    filtered_annotations = [ann for ann in coco_data['annotations'] 
                           if ann['image_id'] in image_ids]
    
    # Create filtered dataset
    filtered_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': coco_data['categories']  # Keep all categories
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f)
    
    return len(filtered_images), len(filtered_annotations)