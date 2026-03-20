import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import random
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile
import os

def create_train_val_split(seed=42):
    """Create 90/10 train/val split at image level with seed=42"""
    print("Creating train/val split...")
    
    # Load original annotations
    with open('data/train/annotations.json', 'r') as f:
        coco_data = json.load(f)
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Get all image IDs
    image_ids = [img['id'] for img in coco_data['images']]
    
    # Try to stratify by store section if possible (from filenames)
    image_sections = []
    for img in coco_data['images']:
        fname = img['file_name'].lower()
        if 'egg' in fname:
            section = 'Egg'
        elif 'frokost' in fname:
            section = 'Frokost'
        elif 'knekkebrod' in fname or 'knekke' in fname:
            section = 'Knekkebrod'
        elif 'varme' in fname or 'drikk' in fname:
            section = 'Varmedrikker'
        else:
            section = 'Other'
        image_sections.append(section)
    
    # Count sections
    section_counts = Counter(image_sections)
    print(f"Section distribution: {dict(section_counts)}")
    
    # If we have good section coverage, stratify; otherwise random split
    if len(section_counts) > 1 and min(section_counts.values()) >= 2:
        print("Using stratified split by store section")
        train_ids, val_ids = train_test_split(
            image_ids, test_size=0.1, random_state=seed, 
            stratify=image_sections
        )
    else:
        print("Using random split (insufficient section diversity)")
        train_ids, val_ids = train_test_split(
            image_ids, test_size=0.1, random_state=seed
        )
    
    print(f"Train images: {len(train_ids)}, Val images: {len(val_ids)}")
    
    # Create train split
    train_images = [img for img in coco_data['images'] if img['id'] in train_ids]
    train_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in train_ids]
    
    train_split = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': coco_data['categories'],
        'info': coco_data.get('info', {})
    }
    
    # Create val split
    val_images = [img for img in coco_data['images'] if img['id'] in val_ids]
    val_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in val_ids]
    
    val_split = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': coco_data['categories'],
        'info': coco_data.get('info', {})
    }
    
    # Save splits
    os.makedirs('data/splits', exist_ok=True)
    
    with open('data/splits/train_split.json', 'w') as f:
        json.dump(train_split, f)
    
    with open('data/splits/val_split.json', 'w') as f:
        json.dump(val_split, f)
    
    print(f"Train annotations: {len(train_annotations)}")
    print(f"Val annotations: {len(val_annotations)}")
    
    return train_split, val_split

def convert_coco_to_yolo(coco_data, output_dir, nc=356, class_mapping=None):
    """Convert COCO format to YOLO txt format"""
    print(f"Converting COCO to YOLO format (nc={nc})...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create images and labels directories
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    # Create image_id to image mapping
    image_map = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image
    image_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_annotations[ann['image_id']].append(ann)
    
    # Convert each image
    for image_id, annotations in image_annotations.items():
        img_info = image_map[image_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Create YOLO label file
        label_file = labels_dir / f"{Path(img_info['file_name']).stem}.txt"
        
        with open(label_file, 'w') as f:
            for ann in annotations:
                # Skip crowd annotations
                if ann.get('iscrowd', 0) == 1:
                    continue
                
                # Get category
                if nc == 1:
                    # Single class detection
                    class_id = 0
                else:
                    # Multi-class
                    class_id = ann['category_id']
                    if class_mapping:
                        class_id = class_mapping.get(class_id, class_id)
                
                # Convert bbox from COCO to YOLO format
                x, y, w, h = ann['bbox']
                
                # COCO: [x_min, y_min, width, height]
                # YOLO: [x_center, y_center, width, height] normalized
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                
                # Write YOLO format line
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    print(f"Converted {len(image_annotations)} images to YOLO format")
    return output_dir

def create_yolo_dataset_yaml(train_dir, val_dir, nc=356, output_path='dataset.yaml'):
    """Create YOLO dataset YAML file"""
    
    # Create class names list
    if nc == 1:
        names = ['product']
    else:
        # Load category names from original data
        with open('data/train/annotations.json', 'r') as f:
            coco_data = json.load(f)
        
        # Create mapping from category_id to name
        cat_names = {}
        for cat in coco_data['categories']:
            cat_names[cat['id']] = cat['name']
        
        # Create ordered list (assuming categories are 0-355)
        names = []
        for i in range(nc):
            names.append(cat_names.get(i, f'class_{i}'))
    
    yaml_content = {
        'path': str(Path.cwd()),  # Dataset root
        'train': str(train_dir),
        'val': str(val_dir),
        'nc': nc,
        'names': names
    }
    
    # Write YAML (using json since yaml import is blocked)
    with open(output_path, 'w') as f:
        f.write(f"path: {yaml_content['path']}\n")
        f.write(f"train: {yaml_content['train']}\n")
        f.write(f"val: {yaml_content['val']}\n")
        f.write(f"nc: {yaml_content['nc']}\n")
        f.write("names:\n")
        for i, name in enumerate(yaml_content['names']):
            f.write(f"  {i}: {name}\n")
    
    print(f"Created YOLO dataset YAML: {output_path}")
    return output_path

def compute_detection_map(gt_coco, pred_coco, iou_threshold=0.5):
    """Compute detection mAP@0.5 (category-agnostic)"""
    # Create temporary COCO objects with all categories mapped to class 0
    
    # Modify ground truth - map all categories to 0
    gt_data = gt_coco.dataset.copy()
    for ann in gt_data['annotations']:
        ann['category_id'] = 0
    gt_data['categories'] = [{'id': 0, 'name': 'product'}]
    
    # Create temporary file for modified GT
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(gt_data, f)
        gt_temp_file = f.name
    
    try:
        # Load modified GT
        gt_coco_det = COCO(gt_temp_file)
        
        # Modify predictions - map all categories to 0
        pred_data = pred_coco.copy()
        for pred in pred_data:
            pred['category_id'] = 0
        
        # Load predictions
        pred_coco_det = gt_coco_det.loadRes(pred_data)
        
        # Evaluate
        coco_eval = COCOeval(gt_coco_det, pred_coco_det, 'bbox')
        coco_eval.params.iouThrs = [iou_threshold]  # Only evaluate at 0.5
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Get mAP@0.5
        detection_map = coco_eval.stats[1]  # mAP@0.5
        
    finally:
        # Clean up temp file
        os.unlink(gt_temp_file)
    
    return detection_map

def compute_classification_map(gt_coco, pred_coco, iou_threshold=0.5):
    """Compute classification mAP@0.5 (with original categories)"""
    # Standard COCO evaluation with original categories
    pred_coco_cls = gt_coco.loadRes(pred_coco)
    
    coco_eval = COCOeval(gt_coco, pred_coco_cls, 'bbox')
    coco_eval.params.iouThrs = [iou_threshold]  # Only evaluate at 0.5
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Get mAP@0.5
    classification_map = coco_eval.stats[1]  # mAP@0.5
    
    return classification_map

def evaluate_predictions(gt_json_path, predictions, iou_threshold=0.5):
    """Evaluate predictions and compute val_score"""
    print(f"Evaluating predictions against {gt_json_path}...")
    
    # Load ground truth
    gt_coco = COCO(gt_json_path)
    
    # Ensure predictions is a list
    if not isinstance(predictions, list):
        predictions = []
    
    if len(predictions) == 0:
        print("No predictions to evaluate")
        return 0.0, 0.0, 0.0
    
    try:
        # Compute detection mAP (category-agnostic)
        detection_map = compute_detection_map(gt_coco, predictions, iou_threshold)
        
        # Compute classification mAP (with categories)
        classification_map = compute_classification_map(gt_coco, predictions, iou_threshold)
        
        # Compute combined score
        val_score = 0.7 * detection_map + 0.3 * classification_map
        
        print(f"Detection mAP@{iou_threshold}: {detection_map:.4f}")
        print(f"Classification mAP@{iou_threshold}: {classification_map:.4f}")
        print(f"Val Score: {val_score:.4f}")
        
        return val_score, detection_map, classification_map
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 0.0, 0.0, 0.0

def test_evaluation_function():
    """Test the evaluation function with dummy predictions"""
    print("Testing evaluation function with dummy predictions...")
    
    # Load val split
    with open('data/splits/val_split.json', 'r') as f:
        val_data = json.load(f)
    
    # Create dummy predictions (random boxes)
    dummy_predictions = []
    for i, img in enumerate(val_data['images'][:3]):  # Test on first 3 images
        # Add a few random predictions per image
        for j in range(3):
            dummy_predictions.append({
                'image_id': img['id'],
                'category_id': random.randint(0, 50),  # Random category
                'bbox': [random.randint(0, 100), random.randint(0, 100), 50, 50],  # Random box
                'score': random.uniform(0.5, 0.9)
            })
    
    # Save val split as temporary file for evaluation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(val_data, f)
        val_temp_file = f.name
    
    try:
        val_score, det_map, cls_map = evaluate_predictions(val_temp_file, dummy_predictions)
        print(f"Dummy evaluation completed: val_score={val_score:.4f}")
        return True
    except Exception as e:
        print(f"Evaluation test failed: {e}")
        return False
    finally:
        os.unlink(val_temp_file)

def main():
    """Main function to create splits, convert data, and test evaluation"""
    print("=== CREATING TRAIN/VAL SPLIT AND YOLO CONVERSION ===")
    
    try:
        # Create train/val split
        train_split, val_split = create_train_val_split(seed=42)
        
        # Convert to YOLO format - multi-class (nc=356)
        train_yolo_mc = convert_coco_to_yolo(train_split, 'data/yolo_mc/train', nc=356)
        val_yolo_mc = convert_coco_to_yolo(val_split, 'data/yolo_mc/val', nc=356)
        
        # Create YOLO dataset YAML for multi-class
        create_yolo_dataset_yaml('data/yolo_mc/train', 'data/yolo_mc/val', nc=356, output_path='data/yolo_mc/dataset.yaml')
        
        # Convert to YOLO format - single class (nc=1)
        train_yolo_sc = convert_coco_to_yolo(train_split, 'data/yolo_sc/train', nc=1)
        val_yolo_sc = convert_coco_to_yolo(val_split, 'data/yolo_sc/val', nc=1)
        
        # Create YOLO dataset YAML for single class
        create_yolo_dataset_yaml('data/yolo_sc/train', 'data/yolo_sc/val', nc=1, output_path='data/yolo_sc/dataset.yaml')
        
        # Test evaluation function
        eval_success = test_evaluation_function()
        
        # Print summary
        print("\n=== SUMMARY ===")
        print(f"Train images: {len(train_split['images'])}")
        print(f"Val images: {len(val_split['images'])}")
        print(f"Train annotations: {len(train_split['annotations'])}")
        print(f"Val annotations: {len(val_split['annotations'])}")
        print(f"Multi-class YOLO data: data/yolo_mc/")
        print(f"Single-class YOLO data: data/yolo_sc/")
        print(f"Evaluation function test: {'PASSED' if eval_success else 'FAILED'}")
        
        # Metrics for orchestrator
        print(f"METRIC:train_images={len(train_split['images'])}")
        print(f"METRIC:val_images={len(val_split['images'])}")
        print(f"METRIC:train_annotations={len(train_split['annotations'])}")
        print(f"METRIC:val_annotations={len(val_split['annotations'])}")
        print(f"METRIC:eval_test_success={1 if eval_success else 0}")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        print("METRIC:setup_success=0")
        return
    
    print("METRIC:setup_success=1")

if __name__ == "__main__":
    main()