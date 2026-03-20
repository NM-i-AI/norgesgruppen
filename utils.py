"""
Shared utilities. Import from main.py as needed.
Keep reusable code here: data loaders, model components, helper functions.
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict

def evaluate_predictions(predictions: List[Dict], val_split: Dict) -> Tuple[float, float, float]:
    """
    Evaluate predictions using COCO metrics.
    
    Args:
        predictions: List of prediction dicts with keys: image_id, category_id, bbox, score
        val_split: COCO format validation split with images, annotations, categories
    
    Returns:
        Tuple of (val_score, detection_mAP@0.5, classification_mAP@0.5)
        where val_score = 0.7 * detection_mAP + 0.3 * classification_mAP
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import tempfile
        import os
    except ImportError as e:
        raise ImportError(f"pycocotools not available: {e}")
    
    # Create temporary files for COCO evaluation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file:
        json.dump(val_split, gt_file)
        gt_path = gt_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as pred_file:
        json.dump(predictions, pred_file)
        pred_path = pred_file.name
    
    try:
        # Load ground truth
        coco_gt = COCO(gt_path)
        
        # Load predictions
        coco_dt = coco_gt.loadRes(pred_path)
        
        # 1. Detection mAP@0.5 (ignore category, treat all as one class)
        # Convert all predictions to category_id=1 for detection-only evaluation
        det_predictions = []
        for pred in predictions:
            det_pred = pred.copy()
            det_pred['category_id'] = 1  # Single class for detection
            det_predictions.append(det_pred)
        
        # Create detection-only ground truth (all annotations become class 1)
        det_gt = val_split.copy()
        det_gt['categories'] = [{'id': 1, 'name': 'product'}]
        det_gt['annotations'] = []
        for ann in val_split['annotations']:
            det_ann = ann.copy()
            det_ann['category_id'] = 1
            det_gt['annotations'].append(det_ann)
        
        # Save detection-only files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as det_gt_file:
            json.dump(det_gt, det_gt_file)
            det_gt_path = det_gt_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as det_pred_file:
            json.dump(det_predictions, det_pred_file)
            det_pred_path = det_pred_file.name
        
        # Evaluate detection
        coco_det_gt = COCO(det_gt_path)
        coco_det_dt = coco_det_gt.loadRes(det_pred_path)
        
        coco_eval_det = COCOeval(coco_det_gt, coco_det_dt, 'bbox')
        coco_eval_det.params.iouThrs = [0.5]  # Only IoU@0.5
        coco_eval_det.evaluate()
        coco_eval_det.accumulate()
        coco_eval_det.summarize()
        
        detection_map = coco_eval_det.stats[0]  # mAP@0.5
        
        # 2. Classification mAP@0.5 (original categories)
        coco_eval_cls = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval_cls.params.iouThrs = [0.5]  # Only IoU@0.5
        coco_eval_cls.evaluate()
        coco_eval_cls.accumulate()
        coco_eval_cls.summarize()
        
        classification_map = coco_eval_cls.stats[0]  # mAP@0.5
        
        # 3. Combined score
        val_score = 0.7 * detection_map + 0.3 * classification_map
        
        return val_score, detection_map, classification_map
        
    finally:
        # Clean up temporary files
        for temp_path in [gt_path, pred_path, det_gt_path, det_pred_path]:
            try:
                os.unlink(temp_path)
            except:
                pass

def convert_coco_to_yolo(coco_json_path: Path, images_dir: Path, output_dir: Path, split_name: str) -> bool:
    """
    Convert COCO format annotations to YOLO format.
    
    Args:
        coco_json_path: Path to COCO JSON file
        images_dir: Directory containing images
        output_dir: Output directory for YOLO format
        split_name: 'train' or 'val'
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load COCO data
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create image_id to filename mapping
        image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        image_id_to_size = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
        
        # Group annotations by image_id
        annotations_by_image = defaultdict(list)
        for ann in coco_data['annotations']:
            annotations_by_image[ann['image_id']].append(ann)
        
        # Convert each image
        converted_count = 0
        for image_id, filename in image_id_to_filename.items():
            # Copy image to output directory
            src_image_path = images_dir / filename
            dst_image_path = output_dir / filename
            
            if src_image_path.exists():
                shutil.copy2(src_image_path, dst_image_path)
                
                # Create corresponding label file
                label_filename = filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
                label_path = output_dir / label_filename
                
                # Get image dimensions
                img_width, img_height = image_id_to_size[image_id]
                
                # Convert annotations to YOLO format
                yolo_labels = []
                for ann in annotations_by_image[image_id]:
                    # COCO bbox format: [x, y, width, height] (top-left corner)
                    # YOLO bbox format: [x_center, y_center, width, height] (normalized)
                    
                    x, y, w, h = ann['bbox']
                    
                    # Convert to center coordinates
                    x_center = x + w / 2
                    y_center = y + h / 2
                    
                    # Normalize by image dimensions
                    x_center_norm = x_center / img_width
                    y_center_norm = y_center / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    
                    # YOLO uses 0-based category IDs, but we keep original category_id
                    category_id = ann['category_id']
                    
                    yolo_labels.append(f"{category_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
                
                # Write label file
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_labels))
                
                converted_count += 1
            else:
                print(f"Warning: Image not found: {src_image_path}")
        
        print(f"Converted {converted_count} images for {split_name} split")
        
        # Create data.yaml file (only once, for the first split)
        data_yaml_path = Path("data/data.yaml")
        if not data_yaml_path.exists():
            # Get category names
            category_names = {}
            for cat in coco_data['categories']:
                category_names[cat['id']] = cat['name']
            
            # Create names list (YOLO expects 0-based indexing)
            max_cat_id = max(category_names.keys())
            names = ['unknown'] * (max_cat_id + 1)
            for cat_id, cat_name in category_names.items():
                names[cat_id] = cat_name
            
            data_yaml_content = {
                'path': str(Path('data').absolute()),
                'train': 'yolo_train',
                'val': 'yolo_val',
                'nc': len(coco_data['categories']),
                'names': names
            }
            
            # Write YAML file (using JSON format since YAML might not be available)
            with open(data_yaml_path, 'w') as f:
                f.write(f"path: {data_yaml_content['path']}\n")
                f.write(f"train: {data_yaml_content['train']}\n")
                f.write(f"val: {data_yaml_content['val']}\n")
                f.write(f"nc: {data_yaml_content['nc']}\n")
                f.write("names:\n")
                for i, name in enumerate(data_yaml_content['names']):
                    f.write(f"  {i}: {name}\n")
            
            print(f"Created data.yaml with {len(names)} categories")
        
        return True
        
    except Exception as e:
        print(f"Error converting {split_name} split: {e}")
        import traceback
        traceback.print_exc()
        return False
