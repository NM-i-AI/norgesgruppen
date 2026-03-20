import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from utils import evaluate_predictions, convert_coco_to_yolo

def create_single_class_annotations(input_json_path: Path, output_json_path: Path):
    """Convert multi-class COCO annotations to single-class (all category_id=1)"""
    print(f"Converting {input_json_path} to single-class format...")
    
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Convert all annotations to category_id=1
    for ann in data['annotations']:
        ann['category_id'] = 1
    
    # Update categories to single class
    data['categories'] = [{'id': 1, 'name': 'product'}]
    
    # Save converted annotations
    with open(output_json_path, 'w') as f:
        json.dump(data, f)
    
    print(f"Converted {len(data['annotations'])} annotations to single class")
    return True

def train_yolov8x_single_class():
    """Train YOLOv8x with nc=1 (detection only) at imgsz=1280"""
    print("=== Training YOLOv8x Single Class (nc=1) ===")
    
    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        print(f"❌ Required packages not available: {e}")
        return None, None, None
    
    # Create single-class versions of train/val splits
    train_single_path = Path("data/train_split_single.json")
    val_single_path = Path("data/val_split_single.json")
    
    if not train_single_path.exists():
        create_single_class_annotations(Path("data/train_split.json"), train_single_path)
    
    if not val_single_path.exists():
        create_single_class_annotations(Path("data/val_split.json"), val_single_path)
    
    # Convert to YOLO format with single class
    yolo_train_single_dir = Path("data/yolo_train_single")
    yolo_val_single_dir = Path("data/yolo_val_single")
    
    if not yolo_train_single_dir.exists():
        train_success = convert_coco_to_yolo(
            coco_json_path=train_single_path,
            images_dir=Path("data/train/images"),
            output_dir=yolo_train_single_dir,
            split_name="train_single"
        )
        if not train_success:
            print("❌ Failed to create single-class train dataset")
            return None, None, None
    
    if not yolo_val_single_dir.exists():
        val_success = convert_coco_to_yolo(
            coco_json_path=val_single_path,
            images_dir=Path("data/train/images"),
            output_dir=yolo_val_single_dir,
            split_name="val_single"
        )
        if not val_success:
            print("❌ Failed to create single-class val dataset")
            return None, None, None
    
    # Create data.yaml for single class
    data_yaml_single_path = Path("data/data_single.yaml")
    if not data_yaml_single_path.exists():
        with open(data_yaml_single_path, 'w') as f:
            f.write(f"path: {Path('data').absolute()}\n")
            f.write("train: yolo_train_single\n")
            f.write("val: yolo_val_single\n")
            f.write("nc: 1\n")
            f.write("names:\n")
            f.write("  0: product\n")
        print(f"Created single-class data.yaml")
    
    print(f"✓ Using single-class YOLO dataset at {data_yaml_single_path}")
    
    # Initialize YOLOv8x model
    print("Initializing YOLOv8x model...")
    model = YOLO('yolov8x.pt')  # Load pretrained weights
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Training parameters for single class
    train_params = {
        'data': str(data_yaml_single_path),
        'epochs': 150,
        'imgsz': 1280,
        'batch': -1,  # Auto batch size
        'device': device,
        'project': 'runs/detect',
        'name': 'yolov8x_single_class',
        'save': True,
        'save_period': 25,  # Save checkpoint every 25 epochs
        'patience': 30,  # Early stopping patience
        'verbose': True,
        'seed': 42,
        # Default augmentation settings
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'close_mosaic': 30  # Close mosaic augmentation at epoch 30
    }
    
    print(f"Training parameters:")
    for key, value in train_params.items():
        print(f"  {key}: {value}")
    
    # Train the model
    print("\nStarting training...")
    try:
        results = model.train(**train_params)
        print("✓ Training completed successfully")
        
        # Get best model path
        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
        print(f"Best model saved at: {best_model_path}")
        
        return results, best_model_path, model
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def evaluate_single_class_model(model_path: Path):
    """Evaluate single-class YOLO model on validation split (detection mAP only)"""
    print(f"\n=== Evaluating Single-Class Model: {model_path} ===")
    
    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        print(f"❌ Required packages not available: {e}")
        return None, None
    
    # Load validation split (original multi-class for evaluation)
    val_split_path = Path("data/val_split.json")
    if not val_split_path.exists():
        print(f"❌ Validation split not found at {val_split_path}")
        return None, None
    
    with open(val_split_path, 'r') as f:
        val_split = json.load(f)
    
    print(f"Validation split: {len(val_split['images'])} images, {len(val_split['annotations'])} annotations")
    
    # Load trained model
    model = YOLO(str(model_path))
    
    # Run inference on validation images
    val_images_dir = Path("data/yolo_val_single")
    val_image_files = list(val_images_dir.glob("*.jpg"))
    
    if not val_image_files:
        print(f"❌ No validation images found in {val_images_dir}")
        return None, None
    
    print(f"Running inference on {len(val_image_files)} validation images...")
    
    # Collect predictions
    predictions = []
    
    for img_path in val_image_files:
        # Extract image_id from filename (e.g., "img_00042.jpg" -> 42)
        filename = img_path.name
        try:
            # Handle different filename formats
            if filename.startswith('img_'):
                image_id = int(filename.split('_')[1].split('.')[0])
            else:
                # Fallback: use the number in filename
                import re
                numbers = re.findall(r'\d+', filename)
                if numbers:
                    image_id = int(numbers[0])
                else:
                    print(f"Warning: Could not extract image_id from {filename}")
                    continue
        except (ValueError, IndexError):
            print(f"Warning: Could not parse image_id from {filename}")
            continue
        
        # Run inference
        results = model(str(img_path), verbose=False)
        
        # Extract predictions (all will be category_id=0 since it's single class)
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, score, cls in zip(boxes, scores, classes):
                    # Convert from x1,y1,x2,y2 to x,y,w,h (COCO format)
                    x1, y1, x2, y2 = box
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    
                    # For single-class model, all predictions are category_id=0
                    predictions.append({
                        'image_id': image_id,
                        'category_id': 0,  # Single class
                        'bbox': [float(x), float(y), float(w), float(h)],
                        'score': float(score)
                    })
    
    print(f"Generated {len(predictions)} predictions")
    
    if not predictions:
        print("❌ No predictions generated")
        return 0.0, 0.0
    
    # For single-class evaluation, we only care about detection mAP
    # Convert ground truth to single class for fair comparison
    det_gt = val_split.copy()
    det_gt['categories'] = [{'id': 0, 'name': 'product'}]
    det_gt['annotations'] = []
    for ann in val_split['annotations']:
        det_ann = ann.copy()
        det_ann['category_id'] = 0  # Convert to single class
        det_gt['annotations'].append(det_ann)
    
    # Evaluate using detection-only evaluation
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import tempfile
        import os
        
        # Create temporary files for COCO evaluation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file:
            json.dump(det_gt, gt_file)
            gt_path = gt_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as pred_file:
            json.dump(predictions, pred_file)
            pred_path = pred_file.name
        
        try:
            # Load ground truth and predictions
            coco_gt = COCO(gt_path)
            coco_dt = coco_gt.loadRes(pred_path)
            
            # Evaluate detection
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.params.iouThrs = [0.5]  # Only IoU@0.5
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            detection_map = coco_eval.stats[0]  # mAP@0.5
            
            print(f"\n=== Single-Class Detection Results ===")
            print(f"Detection mAP@0.5: {detection_map:.4f}")
            print(f"Classification mAP@0.5: 0.0000 (single class)")
            print(f"Combined val_score: {0.7 * detection_map:.4f} (detection only)")
            
            return detection_map, 0.7 * detection_map
            
        finally:
            # Clean up temporary files
            for temp_path in [gt_path, pred_path]:
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_train_val_splits():
    """Create 90/10 train/val splits stratified by image with seed=42"""
    print("=== Creating Train/Val Splits ===")
    
    # Load original annotations
    annotations_path = Path("data/train/annotations.json")
    if not annotations_path.exists():
        print(f"ERROR: Original annotations not found at {annotations_path}")
        return False
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['images'])} images, {len(data['annotations'])} annotations")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Get all image IDs and shuffle them
    image_ids = [img['id'] for img in data['images']]
    random.shuffle(image_ids)
    
    # Calculate split sizes (90/10)
    total_images = len(image_ids)
    val_size = max(1, int(total_images * 0.1))  # At least 1 image in val
    train_size = total_images - val_size
    
    print(f"Split: {train_size} train, {val_size} val ({val_size/total_images:.1%} val)")
    
    # Split image IDs
    val_image_ids = set(image_ids[:val_size])
    train_image_ids = set(image_ids[val_size:])
    
    # Create image lists for each split
    train_images = [img for img in data['images'] if img['id'] in train_image_ids]
    val_images = [img for img in data['images'] if img['id'] in val_image_ids]
    
    # Split annotations by image_id
    train_annotations = [ann for ann in data['annotations'] if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in data['annotations'] if ann['image_id'] in val_image_ids]
    
    print(f"Train: {len(train_images)} images, {len(train_annotations)} annotations")
    print(f"Val: {len(val_images)} images, {len(val_annotations)} annotations")
    
    # Create train split COCO format
    train_split = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': data['categories'],
        'info': data.get('info', {})
    }
    
    # Create val split COCO format
    val_split = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': data['categories'],
        'info': data.get('info', {})
    }
    
    # Save splits
    train_split_path = Path("data/train_split.json")
    val_split_path = Path("data/val_split.json")
    
    with open(train_split_path, 'w') as f:
        json.dump(train_split, f)
    
    with open(val_split_path, 'w') as f:
        json.dump(val_split, f)
    
    print(f"Saved train split to {train_split_path}")
    print(f"Saved val split to {val_split_path}")
    
    return True

def main():
    """Main experiment: Train YOLOv8x with nc=1 (detection only)"""
    print("=== Step 7: YOLOv8x Single Class (nc=1) Training ===")
    
    # Ensure splits exist
    if not Path("data/train_split.json").exists():
        print("Creating train/val splits first...")
        splits_created = create_train_val_splits()
        if not splits_created:
            print("❌ Failed to create splits")
            return
    
    # Train YOLOv8x with single class
    results, best_model_path, model = train_yolov8x_single_class()
    
    if results is None:
        print("❌ Training failed")
        print("METRIC:training_success=0")
        return
    
    print("✓ Training completed successfully")
    print("METRIC:training_success=1")
    
    # Evaluate the trained model
    if best_model_path and best_model_path.exists():
        detection_map, val_score = evaluate_single_class_model(best_model_path)
        
        if detection_map is not None:
            print(f"\n=== Final Results ===")
            print(f"METRIC:detection_map={detection_map:.4f}")
            print(f"METRIC:classification_map=0.0000")
            print(f"METRIC:val_score={val_score:.4f}")
            print(f"METRIC:model_type=single_class")
            print(f"METRIC:num_classes=1")
            print(f"METRIC:model_size=yolov8x")
            print(f"METRIC:image_size=1280")
            print(f"METRIC:epochs=150")
            print(f"METRIC:close_mosaic=30")
            
            # Training metrics from results
            if hasattr(results, 'results_dict'):
                train_metrics = results.results_dict
                if 'metrics/mAP50(B)' in train_metrics:
                    print(f"METRIC:train_map50={train_metrics['metrics/mAP50(B)']:.4f}")
            
            print(f"\n🎉 YOLOv8x single-class complete! Detection mAP@0.5 = {detection_map:.4f}")
            print(f"This is the detection component for two-stage approach.")
        else:
            print("❌ Evaluation failed")
            print("METRIC:evaluation_success=0")
    else:
        print("❌ Best model not found")
        print("METRIC:model_saved=0")

if __name__ == "__main__":
    main()