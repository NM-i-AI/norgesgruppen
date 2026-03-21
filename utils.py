import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile
import os

def evaluate_predictions(predictions, ground_truth_coco):
    """
    Evaluate predictions using COCO metrics and compute val_score.
    
    Args:
        predictions: List of prediction dicts with keys:
            - image_id: int
            - category_id: int  
            - bbox: [x, y, width, height] in COCO format
            - score: float
        ground_truth_coco: COCO format dict with images, annotations, categories
    
    Returns:
        val_score: float (0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5)
        detection_map: float (detection mAP@0.5, category ignored)
        classification_map: float (classification mAP@0.5, category + IoU)
    """
    
    # Create temporary files for COCO evaluation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file:
        json.dump(ground_truth_coco, gt_file)
        gt_path = gt_file.name
    
    try:
        # Load ground truth
        coco_gt = COCO(gt_path)
        
        if not predictions:
            print("Warning: No predictions provided")
            return 0.0, 0.0, 0.0
        
        # Compute detection mAP (category ignored)
        detection_map = compute_detection_map(predictions, coco_gt)
        
        # Compute classification mAP (category + IoU)
        classification_map = compute_classification_map(predictions, coco_gt)
        
        # Compute combined score
        val_score = 0.7 * detection_map + 0.3 * classification_map
        
        return val_score, detection_map, classification_map
        
    finally:
        # Clean up temporary file
        if os.path.exists(gt_path):
            os.unlink(gt_path)

def compute_detection_map(predictions, coco_gt):
    """
    Compute detection mAP@0.5 (category ignored, only IoU matters).
    """
    try:
        # Convert all predictions to category_id=1 for detection-only evaluation
        detection_predictions = []
        for pred in predictions:
            detection_pred = pred.copy()
            detection_pred['category_id'] = 1  # Single category for detection
            detection_predictions.append(detection_pred)
        
        # Create modified ground truth with all categories = 1
        gt_data = coco_gt.dataset.copy()
        
        # Modify categories to have only one category
        gt_data['categories'] = [{'id': 1, 'name': 'object'}]
        
        # Modify annotations to use category_id = 1
        for ann in gt_data['annotations']:
            ann['category_id'] = 1
        
        # Save modified ground truth
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_gt:
            json.dump(gt_data, temp_gt)
            temp_gt_path = temp_gt.name
        
        try:
            # Load modified ground truth
            coco_gt_det = COCO(temp_gt_path)
            
            # Load predictions
            coco_dt = coco_gt_det.loadRes(detection_predictions)
            
            # Evaluate
            coco_eval = COCOeval(coco_gt_det, coco_dt, 'bbox')
            coco_eval.params.iouThrs = [0.5]  # Only IoU@0.5
            coco_eval.params.maxDets = [100, 300, 1000]  # Standard COCO maxDets
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract mAP@0.5
            detection_map = coco_eval.stats[1]  # mAP@0.5
            
        finally:
            if os.path.exists(temp_gt_path):
                os.unlink(temp_gt_path)
        
        return detection_map if not np.isnan(detection_map) else 0.0
        
    except Exception as e:
        print(f"Warning: Detection mAP computation failed: {e}")
        return 0.0

def compute_classification_map(predictions, coco_gt):
    """
    Compute classification mAP@0.5 (category + IoU both matter).
    """
    try:
        # Load predictions with original categories
        coco_dt = coco_gt.loadRes(predictions)
        
        # Evaluate
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.iouThrs = [0.5]  # Only IoU@0.5
        coco_eval.params.maxDets = [100, 300, 1000]  # Standard COCO maxDets
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract mAP@0.5
        classification_map = coco_eval.stats[1]  # mAP@0.5
        
        return classification_map if not np.isnan(classification_map) else 0.0
        
    except Exception as e:
        print(f"Warning: Classification mAP computation failed: {e}")
        return 0.0

def load_coco_split(split_path):
    """
    Load COCO format split file.
    
    Args:
        split_path: Path to COCO JSON file
    
    Returns:
        COCO format dict
    """
    with open(split_path) as f:
        return json.load(f)

def get_image_ids_from_split(split_path):
    """
    Get list of image IDs from a COCO split file.
    
    Args:
        split_path: Path to COCO JSON file
    
    Returns:
        List of image IDs
    """
    coco_data = load_coco_split(split_path)
    return [img['id'] for img in coco_data['images']]

def filter_predictions_by_images(predictions, image_ids):
    """
    Filter predictions to only include specified image IDs.
    
    Args:
        predictions: List of prediction dicts
        image_ids: List of image IDs to keep
    
    Returns:
        Filtered list of predictions
    """
    image_ids_set = set(image_ids)
    return [pred for pred in predictions if pred['image_id'] in image_ids_set]