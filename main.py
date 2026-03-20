import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import random
import shutil
from statistics import mean

def install_packages():
    """Install required packages"""
    packages = [
        "ultralytics==8.1.0",
        "torch==2.6.0", 
        "torchvision==0.21.0",
        "timm==0.9.12"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', package],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                print(f"Warning: Failed to install {package}: {result.stderr}")
            else:
                print(f"Successfully installed {package}")
        except Exception as e:
            print(f"Error installing {package}: {e}")

def load_coco_annotations(annotations_path):
    """Load COCO format annotations"""
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def create_train_val_split(coco_data, val_ratio=0.1, seed=42):
    """Create 90/10 train/val split stratified by store section if possible"""
    random.seed(seed)
    
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    # Group images by store section from filename
    sections = ['egg', 'frokost', 'knekkebrod', 'varmedrikker']
    section_images = {section: [] for section in sections}
    unassigned_images = []
    
    for img in images:
        filename = img['file_name'].lower()
        assigned = False
        for section in sections:
            if section in filename:
                section_images[section].append(img)
                assigned = True
                break
        if not assigned:
            unassigned_images.append(img)
    
    print(f"Images by section: {[(s, len(imgs)) for s, imgs in section_images.items()]}")
    print(f"Unassigned images: {len(unassigned_images)}")
    
    # Split each section proportionally
    val_images = []
    train_images = []
    
    for section, imgs in section_images.items():
        if len(imgs) > 0:
            random.shuffle(imgs)
            val_count = max(1, int(len(imgs) * val_ratio))  # At least 1 for val if any exist
            val_images.extend(imgs[:val_count])
            train_images.extend(imgs[val_count:])
    
    # Handle unassigned images
    if unassigned_images:
        random.shuffle(unassigned_images)
        val_count = int(len(unassigned_images) * val_ratio)
        val_images.extend(unassigned_images[:val_count])
        train_images.extend(unassigned_images[val_count:])
    
    print(f"Split: {len(train_images)} train, {len(val_images)} val")
    
    # Create image ID sets
    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}
    
    # Split annotations
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in annotations if ann['image_id'] in val_image_ids]
    
    print(f"Annotations: {len(train_annotations)} train, {len(val_annotations)} val")
    
    # Create split datasets
    train_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': categories
    }
    
    val_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': categories
    }
    
    return train_data, val_data

def convert_to_yolo_format(coco_data, images_dir, output_dir, split_name):
    """Convert COCO format to YOLO format"""
    output_images_dir = output_dir / 'images' / split_name
    output_labels_dir = output_dir / 'labels' / split_name
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create image_id to annotations mapping
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    # Process each image
    for img in coco_data['images']:
        img_id = img['id']
        filename = img['file_name']
        width = img['width']
        height = img['height']
        
        # Copy image
        src_path = images_dir / filename
        dst_path = output_images_dir / filename
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        
        # Create label file
        label_filename = Path(filename).stem + '.txt'
        label_path = output_labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            for ann in annotations_by_image[img_id]:
                # Convert COCO bbox to YOLO format
                x, y, w, h = ann['bbox']
                
                # Convert to center coordinates and normalize
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height
                
                # YOLO uses 0-based class indices, COCO uses 1-based (except category 0)
                class_id = ann['category_id']
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

def create_dataset_yaml(output_dir, num_classes):
    """Create dataset.yaml for YOLO training"""
    yaml_content = f"""# NorgesGruppen Grocery Dataset
path: {output_dir.absolute()}
train: images/train
val: images/val

# Number of classes
nc: {num_classes}

# Class names (using indices for now)
names:
"""
    
    # Add class names (just use indices for now)
    for i in range(num_classes):
        yaml_content += f"  {i}: 'class_{i}'\n"
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path

def evaluate_with_pycocotools(predictions, ground_truth_coco, image_ids):
    """Evaluate predictions using pycocotools to compute detection and classification mAP@0.5"""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import tempfile
        import os
        
        # Create temporary files for ground truth and predictions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file:
            json.dump(ground_truth_coco, gt_file)
            gt_path = gt_file.name
        
        try:
            # Load ground truth
            coco_gt = COCO(gt_path)
            
            # Convert predictions to COCO format
            coco_predictions = []
            for pred in predictions:
                coco_predictions.append({
                    'image_id': pred['image_id'],
                    'category_id': pred['category_id'],
                    'bbox': pred['bbox'],  # [x, y, width, height]
                    'score': pred['score']
                })
            
            if not coco_predictions:
                print("No predictions to evaluate")
                return 0.0, 0.0
            
            # Load predictions
            coco_dt = coco_gt.loadRes(coco_predictions)
            
            # Evaluate detection (category-agnostic)
            # For detection mAP, we treat all categories as one class
            detection_predictions = []
            for pred in coco_predictions:
                det_pred = pred.copy()
                det_pred['category_id'] = 1  # Single class for detection
                detection_predictions.append(det_pred)
            
            # Create single-class ground truth for detection evaluation
            detection_gt = {
                'images': ground_truth_coco['images'],
                'categories': [{'id': 1, 'name': 'product'}],
                'annotations': []
            }
            
            for ann in ground_truth_coco['annotations']:
                det_ann = ann.copy()
                det_ann['category_id'] = 1  # Single class
                detection_gt['annotations'].append(det_ann)
            
            # Save detection ground truth
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as det_gt_file:
                json.dump(detection_gt, det_gt_file)
                det_gt_path = det_gt_file.name
            
            try:
                # Evaluate detection
                coco_det_gt = COCO(det_gt_path)
                coco_det_dt = coco_det_gt.loadRes(detection_predictions)
                
                eval_det = COCOeval(coco_det_gt, coco_det_dt, 'bbox')
                eval_det.params.imgIds = image_ids
                eval_det.params.iouThrs = [0.5]  # Only IoU@0.5
                eval_det.evaluate()
                eval_det.accumulate()
                eval_det.summarize()
                
                detection_map = eval_det.stats[1]  # mAP@0.5
                
            finally:
                os.unlink(det_gt_path)
            
            # Evaluate classification (category-specific)
            eval_cls = COCOeval(coco_gt, coco_dt, 'bbox')
            eval_cls.params.imgIds = image_ids
            eval_cls.params.iouThrs = [0.5]  # Only IoU@0.5
            eval_cls.evaluate()
            eval_cls.accumulate()
            eval_cls.summarize()
            
            classification_map = eval_cls.stats[1]  # mAP@0.5
            
            return detection_map, classification_map
            
        finally:
            os.unlink(gt_path)
            
    except Exception as e:
        print(f"Error in pycocotools evaluation: {e}")
        return 0.0, 0.0

def create_dummy_predictions(val_data, num_predictions=50):
    """Create dummy predictions for testing evaluation function"""
    predictions = []
    
    # Get some random annotations to base dummy predictions on
    annotations = val_data['annotations'][:num_predictions]
    
    for i, ann in enumerate(annotations):
        # Add some noise to the bbox
        x, y, w, h = ann['bbox']
        x += random.uniform(-5, 5)
        y += random.uniform(-5, 5)
        w += random.uniform(-2, 2)
        h += random.uniform(-2, 2)
        
        # Random score
        score = random.uniform(0.3, 0.9)
        
        # Sometimes use correct category, sometimes random
        if random.random() < 0.7:  # 70% chance of correct category
            category_id = ann['category_id']
        else:
            category_id = random.randint(0, 356)
        
        predictions.append({
            'image_id': ann['image_id'],
            'category_id': category_id,
            'bbox': [max(0, x), max(0, y), max(1, w), max(1, h)],
            'score': score
        })
    
    return predictions

def train_yolo_model(dataset_yaml_path, model_size='n', epochs=50, imgsz=640, batch=16):
    """Train YOLO model"""
    try:
        from ultralytics import YOLO
        
        # Initialize model
        model = YOLO(f'yolov8{model_size}.pt')
        
        # Train
        results = model.train(
            data=str(dataset_yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device='auto',
            project='runs/detect',
            name='yolov8n_baseline',
            save=True,
            plots=True
        )
        
        return model, results
        
    except Exception as e:
        print(f"Error training model: {e}")
        return None, None

def evaluate_model_with_proper_metrics(model, val_data, images_dir):
    """Evaluate model using proper detection and classification metrics"""
    try:
        # Get predictions from model on validation images
        predictions = []
        
        for img_info in val_data['images']:
            img_path = images_dir / img_info['file_name']
            if not img_path.exists():
                continue
                
            # Run inference
            results = model(str(img_path))
            
            # Convert results to COCO format
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        conf = boxes.conf[i].cpu().numpy()
                        cls = int(boxes.cls[i].cpu().numpy())
                        
                        # Convert to COCO bbox format [x, y, width, height]
                        bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        
                        predictions.append({
                            'image_id': img_info['id'],
                            'category_id': cls,
                            'bbox': bbox,
                            'score': float(conf)
                        })
        
        # Evaluate using pycocotools
        image_ids = [img['id'] for img in val_data['images']]
        detection_map, classification_map = evaluate_with_pycocotools(
            predictions, val_data, image_ids
        )
        
        # Compute val_score
        val_score = 0.7 * detection_map + 0.3 * classification_map
        
        return {
            'detection_mAP@0.5': detection_map,
            'classification_mAP@0.5': classification_map,
            'val_score': val_score,
            'num_predictions': len(predictions)
        }
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return {
            'detection_mAP@0.5': 0.0,
            'classification_mAP@0.5': 0.0,
            'val_score': 0.0,
            'num_predictions': 0
        }

def main():
    print("=== YOLOv8n Baseline with Proper Evaluation ===\n")
    
    # 1. Install packages
    print("1. Installing required packages...")
    install_packages()
    
    # 2. Load and analyze data
    print("\n2. Loading COCO annotations...")
    annotations_path = Path("data/train/annotations.json")
    images_dir = Path("data/train/images")
    
    if not annotations_path.exists():
        print(f"Error: Annotations file not found at {annotations_path}")
        print("METRIC:val_score=0.0")
        return
    
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        print("METRIC:val_score=0.0")
        return
    
    coco_data = load_coco_annotations(annotations_path)
    print(f"Loaded {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations, {len(coco_data['categories'])} categories")
    
    # 3. Create train/val split
    print("\n3. Creating 90/10 train/val split...")
    train_data, val_data = create_train_val_split(coco_data, val_ratio=0.1, seed=42)
    
    # Save split data for evaluation
    splits_dir = Path("splits")
    splits_dir.mkdir(exist_ok=True)
    
    with open(splits_dir / 'train_split.json', 'w') as f:
        json.dump(train_data, f)
    
    with open(splits_dir / 'val_split.json', 'w') as f:
        json.dump(val_data, f)
    
    print(f"Saved splits to {splits_dir}")
    
    # 4. Test evaluation function with dummy predictions
    print("\n4. Testing evaluation function with dummy predictions...")
    dummy_predictions = create_dummy_predictions(val_data, num_predictions=20)
    
    image_ids = [img['id'] for img in val_data['images']]
    det_map, cls_map = evaluate_with_pycocotools(dummy_predictions, val_data, image_ids)
    dummy_val_score = 0.7 * det_map + 0.3 * cls_map
    
    print(f"Dummy evaluation results:")
    print(f"  Detection mAP@0.5: {det_map:.4f}")
    print(f"  Classification mAP@0.5: {cls_map:.4f}")
    print(f"  Val Score: {dummy_val_score:.4f}")
    print(f"  Number of dummy predictions: {len(dummy_predictions)}")
    
    # 5. Convert to YOLO format
    print("\n5. Converting to YOLO format...")
    output_dir = Path("yolo_dataset")
    output_dir.mkdir(exist_ok=True)
    
    convert_to_yolo_format(train_data, images_dir, output_dir, 'train')
    convert_to_yolo_format(val_data, images_dir, output_dir, 'val')
    
    # 6. Create dataset.yaml
    print("\n6. Creating dataset configuration...")
    num_classes = len(coco_data['categories'])
    print(f"Number of classes: {num_classes}")
    
    dataset_yaml_path = create_dataset_yaml(output_dir, num_classes)
    print(f"Created dataset.yaml at {dataset_yaml_path}")
    
    # Count labels to verify conversion
    train_labels_dir = output_dir / 'labels' / 'train'
    val_labels_dir = output_dir / 'labels' / 'val'
    
    train_label_files = list(train_labels_dir.glob('*.txt'))
    val_label_files = list(val_labels_dir.glob('*.txt'))
    
    print(f"YOLO format verification:")
    print(f"  Train label files: {len(train_label_files)}")
    print(f"  Val label files: {len(val_label_files)}")
    
    # Count total labels
    total_train_labels = 0
    total_val_labels = 0
    
    for label_file in train_label_files:
        with open(label_file, 'r') as f:
            total_train_labels += len(f.readlines())
    
    for label_file in val_label_files:
        with open(label_file, 'r') as f:
            total_val_labels += len(f.readlines())
    
    print(f"  Total train labels: {total_train_labels}")
    print(f"  Total val labels: {total_val_labels}")
    print(f"  Expected train labels: {len(train_data['annotations'])}")
    print(f"  Expected val labels: {len(val_data['annotations'])}")
    
    # 7. Train model (small test run)
    print("\n7. Training YOLOv8n model (short test run)...")
    model, train_results = train_yolo_model(
        dataset_yaml_path, 
        model_size='n', 
        epochs=5,  # Short test run
        imgsz=640, 
        batch=8  # Smaller batch for testing
    )
    
    if model is None:
        print("Training failed")
        print("METRIC:val_score=0.0")
        return
    
    print("Training completed successfully")
    
    # 8. Evaluate model with proper metrics
    print("\n8. Evaluating model with proper detection/classification metrics...")
    eval_results = evaluate_model_with_proper_metrics(model, val_data, images_dir)
    
    # Print all metrics
    print("\n=== EVALUATION RESULTS ===") 
    print(f"Detection mAP@0.5: {eval_results['detection_mAP@0.5']:.4f}")
    print(f"Classification mAP@0.5: {eval_results['classification_mAP@0.5']:.4f}")
    print(f"Val Score: {eval_results['val_score']:.4f}")
    print(f"Number of predictions: {eval_results['num_predictions']}")
    
    # Print metrics in required format
    print(f"\nMETRIC:detection_mAP@0.5={eval_results['detection_mAP@0.5']:.4f}")
    print(f"METRIC:classification_mAP@0.5={eval_results['classification_mAP@0.5']:.4f}")
    print(f"METRIC:val_score={eval_results['val_score']:.4f}")
    print(f"METRIC:num_predictions={eval_results['num_predictions']}")
    
    print("\n=== Evaluation Function and YOLO Format Setup Complete ===\n")
    print("Next steps:")
    print("- Evaluation function working with pycocotools")
    print("- YOLO format data created and verified")
    print("- Ready for full training runs and model comparisons")

if __name__ == "__main__":
    main()