import json
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
import shutil

def create_train_val_split(annotations_path='data/train/annotations.json', 
                          train_output='train_split.json',
                          val_output='val_split.json',
                          val_ratio=0.1, 
                          seed=42):
    """
    Create 90/10 stratified train/val split by images, preserving store section proportions.
    
    Args:
        annotations_path: Path to original COCO annotations
        train_output: Output path for train split
        val_output: Output path for val split  
        val_ratio: Fraction of images for validation (default 0.1 = 10%)
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_image_count, val_image_count)
    """
    np.random.seed(seed)
    
    # Load original annotations
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Group images by store section
    section_images = defaultdict(list)
    for img in coco_data['images']:
        filename = img['file_name']
        # Extract store section from filename
        if 'Egg' in filename:
            section = 'Egg'
        elif 'Frokost' in filename:
            section = 'Frokost'
        elif 'Knekkebrod' in filename:
            section = 'Knekkebrod'
        elif 'Varmedrikker' in filename:
            section = 'Varmedrikker'
        else:
            section = 'Unknown'
        section_images[section].append(img)
    
    print(f"Images per section: {dict((k, len(v)) for k, v in section_images.items())}")
    
    # Stratified sampling from each section
    val_images = []
    train_images = []
    
    for section, images in section_images.items():
        # Shuffle images in this section
        section_imgs = images.copy()
        np.random.shuffle(section_imgs)
        
        # Calculate val count for this section
        section_val_count = max(1, int(len(section_imgs) * val_ratio))
        
        # Split
        val_images.extend(section_imgs[:section_val_count])
        train_images.extend(section_imgs[section_val_count:])
        
        print(f"Section {section}: {len(section_imgs)} total -> {section_val_count} val, {len(section_imgs) - section_val_count} train")
    
    # Create image ID sets for filtering annotations
    val_image_ids = {img['id'] for img in val_images}
    train_image_ids = {img['id'] for img in train_images}
    
    # Filter annotations
    train_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in val_image_ids]
    
    # Create train split
    train_split = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': coco_data['categories'],
        'info': coco_data.get('info', {})
    }
    
    # Create val split
    val_split = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': coco_data['categories'],
        'info': coco_data.get('info', {})
    }
    
    # Save splits
    with open(train_output, 'w') as f:
        json.dump(train_split, f)
    
    with open(val_output, 'w') as f:
        json.dump(val_split, f)
    
    print(f"\nSplit created:")
    print(f"  Train: {len(train_images)} images, {len(train_annotations)} annotations")
    print(f"  Val: {len(val_images)} images, {len(val_annotations)} annotations")
    print(f"  Ratio: {len(val_images) / (len(train_images) + len(val_images)):.1%} validation")
    
    return len(train_images), len(val_images)

def create_yolo_labels(coco_json_path, output_dir, image_dir='data/train/images'):
    """
    Convert COCO annotations to YOLO format labels.
    
    Args:
        coco_json_path: Path to COCO JSON file
        output_dir: Directory to save YOLO label files
        image_dir: Directory containing images (for getting dimensions)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load COCO data
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to filename mapping
    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    id_to_dims = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
    
    # Group annotations by image
    image_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_annotations[ann['image_id']].append(ann)
    
    # Convert each image's annotations
    for image_id, annotations in image_annotations.items():
        filename = id_to_filename[image_id]
        img_width, img_height = id_to_dims[image_id]
        
        # Create label file
        label_filename = Path(filename).stem + '.txt'
        label_path = output_dir / label_filename
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # Convert COCO bbox [x, y, width, height] to YOLO [x_center, y_center, width, height] normalized
                x, y, w, h = ann['bbox']
                x_center = (x + w/2) / img_width
                y_center = (y + h/2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                
                # YOLO format: class_id x_center y_center width height
                f.write(f"{ann['category_id']} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    print(f"Created YOLO labels for {len(image_annotations)} images in {output_dir}")

def yolo_predictions_to_coco(predictions, image_info, score_threshold=0.01):
    """
    Convert YOLO model predictions to COCO format.
    
    Args:
        predictions: YOLO model output (list of tensors or Results objects)
        image_info: List of dicts with 'id', 'width', 'height' for each image
        score_threshold: Minimum confidence to include
    
    Returns:
        List of COCO-format predictions
    """
    coco_predictions = []
    
    for i, (pred, img_info) in enumerate(zip(predictions, image_info)):
        image_id = img_info['id']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Handle different YOLO output formats
        if hasattr(pred, 'boxes'):  # ultralytics Results object
            boxes = pred.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                
                for j in range(len(xyxy)):
                    if conf[j] >= score_threshold:
                        x1, y1, x2, y2 = xyxy[j]
                        # Convert to COCO format [x, y, width, height]
                        bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        
                        coco_predictions.append({
                            'image_id': int(image_id),
                            'category_id': int(cls[j]),
                            'bbox': bbox,
                            'score': float(conf[j])
                        })
        else:
            # Handle tensor format if needed
            # This would need to be implemented based on specific tensor format
            pass
    
    return coco_predictions

def evaluate_coco_predictions(gt_json_path, predictions, verbose=True):
    """
    Evaluate predictions using COCO metrics.
    
    Args:
        gt_json_path: Path to ground truth COCO JSON
        predictions: List of COCO-format predictions
        verbose: Whether to print detailed results
    
    Returns:
        dict: Contains detection_mAP, classification_mAP, and val_score
    """
    # Load ground truth
    coco_gt = COCO(gt_json_path)
    
    if len(predictions) == 0:
        print("Warning: No predictions to evaluate")
        return {
            'detection_mAP': 0.0,
            'classification_mAP': 0.0,
            'val_score': 0.0
        }
    
    # Create temporary predictions file
    pred_file = 'temp_predictions.json'
    with open(pred_file, 'w') as f:
        json.dump(predictions, f)
    
    try:
        # Load predictions
        coco_dt = coco_gt.loadRes(pred_file)
        
        # 1. Detection mAP@0.5 (category-agnostic)
        # Map all categories to class 0 for detection evaluation
        detection_predictions = []
        for pred in predictions:
            det_pred = pred.copy()
            det_pred['category_id'] = 0  # Map all to single class
            detection_predictions.append(det_pred)
        
        # Save detection predictions
        det_pred_file = 'temp_detection_predictions.json'
        with open(det_pred_file, 'w') as f:
            json.dump(detection_predictions, f)
        
        # Create single-class ground truth
        detection_gt_data = {
            'images': coco_gt.dataset['images'],
            'categories': [{'id': 0, 'name': 'product'}],
            'annotations': []
        }
        
        for ann in coco_gt.dataset['annotations']:
            det_ann = ann.copy()
            det_ann['category_id'] = 0
            detection_gt_data['annotations'].append(det_ann)
        
        det_gt_file = 'temp_detection_gt.json'
        with open(det_gt_file, 'w') as f:
            json.dump(detection_gt_data, f)
        
        # Evaluate detection
        coco_det_gt = COCO(det_gt_file)
        coco_det_dt = coco_det_gt.loadRes(det_pred_file)
        
        coco_eval_det = COCOeval(coco_det_gt, coco_det_dt, 'bbox')
        coco_eval_det.params.iouThrs = [0.5]  # Only IoU@0.5
        coco_eval_det.evaluate()
        coco_eval_det.accumulate()
        coco_eval_det.summarize()
        
        detection_mAP = coco_eval_det.stats[0]  # mAP@0.5
        
        # 2. Classification mAP@0.5 (with original categories)
        coco_eval_cls = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval_cls.params.iouThrs = [0.5]  # Only IoU@0.5
        coco_eval_cls.evaluate()
        coco_eval_cls.accumulate()
        coco_eval_cls.summarize()
        
        classification_mAP = coco_eval_cls.stats[0]  # mAP@0.5
        
        # 3. Combined score
        val_score = 0.7 * detection_mAP + 0.3 * classification_mAP
        
        if verbose:
            print(f"\nEvaluation Results:")
            print(f"  Detection mAP@0.5: {detection_mAP:.4f}")
            print(f"  Classification mAP@0.5: {classification_mAP:.4f}")
            print(f"  Val Score (0.7*det + 0.3*cls): {val_score:.4f}")
        
        # Cleanup temp files
        Path(pred_file).unlink(missing_ok=True)
        Path(det_pred_file).unlink(missing_ok=True)
        Path(det_gt_file).unlink(missing_ok=True)
        
        return {
            'detection_mAP': detection_mAP,
            'classification_mAP': classification_mAP,
            'val_score': val_score
        }
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        # Cleanup temp files
        Path(pred_file).unlink(missing_ok=True)
        Path(det_pred_file).unlink(missing_ok=True)
        Path(det_gt_file).unlink(missing_ok=True)
        
        return {
            'detection_mAP': 0.0,
            'classification_mAP': 0.0,
            'val_score': 0.0
        }

def setup_yolo_dataset_structure():
    """
    Create YOLO dataset directory structure and copy images.
    """
    # Create directories
    train_img_dir = Path('datasets/train/images')
    train_lbl_dir = Path('datasets/train/labels')
    val_img_dir = Path('datasets/val/images')
    val_lbl_dir = Path('datasets/val/labels')
    
    for dir_path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load splits to get image lists
    with open('train_split.json', 'r') as f:
        train_data = json.load(f)
    with open('val_split.json', 'r') as f:
        val_data = json.load(f)
    
    # Copy train images
    source_img_dir = Path('data/train/images')
    for img_info in train_data['images']:
        src = source_img_dir / img_info['file_name']
        dst = train_img_dir / img_info['file_name']
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    
    # Copy val images
    for img_info in val_data['images']:
        src = source_img_dir / img_info['file_name']
        dst = val_img_dir / img_info['file_name']
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    
    # Create YOLO labels
    create_yolo_labels('train_split.json', train_lbl_dir)
    create_yolo_labels('val_split.json', val_lbl_dir)
    
    print(f"YOLO dataset structure created:")
    print(f"  Train: {len(list(train_img_dir.glob('*.jpg')))} images, {len(list(train_lbl_dir.glob('*.txt')))} labels")
    print(f"  Val: {len(list(val_img_dir.glob('*.jpg')))} images, {len(list(val_lbl_dir.glob('*.txt')))} labels")

def test_evaluation_pipeline():
    """
    Test the evaluation pipeline with dummy predictions.
    """
    print("\n=== Testing Evaluation Pipeline ===")
    
    # Load val split to get image info
    with open('val_split.json', 'r') as f:
        val_data = json.load(f)
    
    # Create dummy predictions (random boxes)
    np.random.seed(42)
    dummy_predictions = []
    
    for img_info in val_data['images'][:3]:  # Test on first 3 images
        image_id = img_info['id']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Generate 2-5 random boxes per image
        num_boxes = np.random.randint(2, 6)
        for _ in range(num_boxes):
            # Random box
            x = np.random.uniform(0, img_width * 0.8)
            y = np.random.uniform(0, img_height * 0.8)
            w = np.random.uniform(50, img_width * 0.2)
            h = np.random.uniform(50, img_height * 0.2)
            
            dummy_predictions.append({
                'image_id': image_id,
                'category_id': np.random.randint(0, 357),
                'bbox': [x, y, w, h],
                'score': np.random.uniform(0.3, 0.9)
            })
    
    print(f"Created {len(dummy_predictions)} dummy predictions")
    
    # Test evaluation
    results = evaluate_coco_predictions('val_split.json', dummy_predictions)
    
    print(f"\nDummy evaluation results:")
    print(f"  Detection mAP@0.5: {results['detection_mAP']:.4f}")
    print(f"  Classification mAP@0.5: {results['classification_mAP']:.4f}")
    print(f"  Val Score: {results['val_score']:.4f}")
    
    return results
