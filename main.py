import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import random
from utils import evaluate_predictions, load_coco_split
from ultralytics import YOLO
import torch

def train_yolov8x_1280():
    """Train YOLOv8x with 356 classes at 1280px resolution."""
    print("=== Training YOLOv8x at 1280px ===")
    
    # Check if splits exist
    data_path = Path("data")
    train_split_path = data_path / "train_split.json"
    val_split_path = data_path / "val_split.json"
    
    if not train_split_path.exists() or not val_split_path.exists():
        print("Error: Train/val splits not found. Run data preparation first.")
        return
    
    # Load splits to get image lists
    train_coco = load_coco_split(train_split_path)
    val_coco = load_coco_split(val_split_path)
    
    train_image_ids = set(img['id'] for img in train_coco['images'])
    val_image_ids = set(img['id'] for img in val_coco['images'])
    
    print(f"Train images: {len(train_image_ids)}")
    print(f"Val images: {len(val_image_ids)}")
    print(f"Train annotations: {len(train_coco['annotations'])}")
    print(f"Val annotations: {len(val_coco['annotations'])}")
    
    # Count categories in training data
    train_categories = set(ann['category_id'] for ann in train_coco['annotations'])
    print(f"Categories in training data: {len(train_categories)} (range: {min(train_categories)}-{max(train_categories)})")
    
    # Create custom data.yaml for this experiment
    yaml_content = f"""# YOLOv8x 1280px experiment
path: {data_path.absolute()}
train: train/images
val: train/images

nc: {len(train_categories)}
names: {list(range(len(train_categories)))}
"""
    
    yaml_path = data_path / "yolov8x_1280.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    # Create filtered YOLO labels for this split
    labels_dir = data_path / "labels_yolov8x_1280"
    labels_dir.mkdir(exist_ok=True)
    
    # Convert train split to YOLO labels
    train_labels_dir = labels_dir / "train"
    train_labels_dir.mkdir(exist_ok=True)
    
    val_labels_dir = labels_dir / "val" 
    val_labels_dir.mkdir(exist_ok=True)
    
    # Create image_id to filename mapping
    id_to_filename = {img['id']: img['file_name'] for img in train_coco['images'] + val_coco['images']}
    id_to_size = {img['id']: (img['width'], img['height']) for img in train_coco['images'] + val_coco['images']}
    
    # Convert train annotations
    train_image_annotations = defaultdict(list)
    for ann in train_coco['annotations']:
        train_image_annotations[ann['image_id']].append(ann)
    
    for image_id, annotations in train_image_annotations.items():
        filename = id_to_filename[image_id]
        img_w, img_h = id_to_size[image_id]
        
        label_filename = Path(filename).stem + '.txt'
        label_path = train_labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                x, y, w, h = ann['bbox']
                x_center = (x + w/2) / img_w
                y_center = (y + h/2) / img_h
                width = w / img_w
                height = h / img_h
                class_id = ann['category_id']
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Convert val annotations
    val_image_annotations = defaultdict(list)
    for ann in val_coco['annotations']:
        val_image_annotations[ann['image_id']].append(ann)
    
    for image_id, annotations in val_image_annotations.items():
        filename = id_to_filename[image_id]
        img_w, img_h = id_to_size[image_id]
        
        label_filename = Path(filename).stem + '.txt'
        label_path = val_labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                x, y, w, h = ann['bbox']
                x_center = (x + w/2) / img_w
                y_center = (y + h/2) / img_h
                width = w / img_w
                height = h / img_h
                class_id = ann['category_id']
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"Created YOLO labels in {labels_dir}")
    
    # Update yaml to point to our custom labels
    yaml_content = f"""# YOLOv8x 1280px experiment
path: {data_path.absolute()}
train: labels_yolov8x_1280/train
val: labels_yolov8x_1280/val

nc: {len(train_categories)}
names: {list(range(len(train_categories)))}
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    # Initialize YOLOv8x model with weights_only=False workaround
    try:
        # Try to set torch.load to use weights_only=False globally
        import torch.serialization
        original_load = torch.load
        
        def patched_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        
        model = YOLO('yolov8x.pt')
        
        # Restore original torch.load
        torch.load = original_load
        
    except Exception as e:
        print(f"Error loading YOLOv8x model: {e}")
        print("Trying alternative approach...")
        
        # Alternative: try creating model from scratch
        try:
            model = YOLO('yolov8x.yaml')  # Load architecture only
        except Exception as e2:
            print(f"Failed to create model from yaml: {e2}")
            print("METRIC:val_score=0.0000")
            print("METRIC:training_failed=1")
            return
    
    # Auto-adjust batch size based on GPU memory
    # YOLOv8x at 1280px is memory intensive
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"GPU memory: {gpu_memory:.1f} GB")
        
        # Conservative batch size for YOLOv8x at 1280px
        if gpu_memory >= 80:  # A800 80GB
            batch_size = 8
        elif gpu_memory >= 40:
            batch_size = 4
        elif gpu_memory >= 24:
            batch_size = 2
        else:
            batch_size = 1
    else:
        batch_size = 1
    
    print(f"\nStarting training...")
    print(f"Model: YOLOv8x")
    print(f"Classes: {len(train_categories)}")
    print(f"Image size: 1280")
    print(f"Epochs: 100")
    print(f"Batch size: {batch_size}")
    
    # Train the model
    try:
        results = model.train(
            data=str(yaml_path),
            epochs=100,
            imgsz=1280,
            batch=batch_size,
            device='0',  # Use first GPU
            project='runs/detect',
            name='yolov8x_1280',
            save=True,
            save_period=25,  # Save every 25 epochs
            patience=40,  # Early stopping patience
            verbose=True,
            close_mosaic=50,  # Close mosaic augmentation after 50 epochs
            amp=True,  # Automatic mixed precision
            workers=8  # Data loading workers
        )
        
        print(f"\nTraining completed successfully!")
        
        # Load best model for evaluation
        best_model_path = Path('runs/detect/yolov8x_1280/weights/best.pt')
        if best_model_path.exists():
            # Use the same patched loading for the trained model
            try:
                import torch.serialization
                original_load = torch.load
                
                def patched_load(*args, **kwargs):
                    if 'weights_only' not in kwargs:
                        kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                torch.load = patched_load
                model = YOLO(str(best_model_path))
                torch.load = original_load
                
                print(f"Loaded best model from {best_model_path}")
            except Exception as e:
                print(f"Warning: Could not load best model: {e}, using last model")
        else:
            print("Warning: Best model not found, using last model")
        
        # Run inference on validation set
        print(f"\nRunning inference on validation set...")
        
        # Get validation image paths
        val_image_paths = []
        images_dir = data_path / "train" / "images"
        
        for img_info in val_coco['images']:
            img_path = images_dir / img_info['file_name']
            if img_path.exists():
                val_image_paths.append(str(img_path))
        
        print(f"Found {len(val_image_paths)} validation images")
        
        # Run inference
        predictions = []
        
        for img_path in val_image_paths:
            # Extract image_id from filename
            img_filename = Path(img_path).name
            # Assuming format like img_00001.jpg -> image_id = 1
            try:
                if img_filename.startswith('img_'):
                    image_id = int(img_filename.split('_')[1].split('.')[0])
                else:
                    # Fallback: use filename without extension as ID
                    image_id = int(Path(img_filename).stem)
            except:
                print(f"Warning: Could not extract image_id from {img_filename}")
                continue
            
            # Run inference with higher confidence threshold for cleaner results
            results = model(img_path, verbose=False, conf=0.1, iou=0.7)
            
            # Convert results to COCO format
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # Get box coordinates in xyxy format
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        conf = boxes.conf[i].cpu().numpy()
                        cls = int(boxes.cls[i].cpu().numpy())
                        
                        # Convert xyxy to xywh (COCO format)
                        x1, y1, x2, y2 = xyxy
                        x, y, w, h = x1, y1, x2-x1, y2-y1
                        
                        predictions.append({
                            'image_id': image_id,
                            'category_id': cls,
                            'bbox': [float(x), float(y), float(w), float(h)],
                            'score': float(conf)
                        })
        
        print(f"Generated {len(predictions)} predictions")
        
        # Evaluate predictions
        if predictions:
            val_score, det_map, cls_map = evaluate_predictions(predictions, val_coco)
            
            print(f"\n=== Evaluation Results ===")
            print(f"Detection mAP@0.5: {det_map:.4f}")
            print(f"Classification mAP@0.5: {cls_map:.4f}")
            print(f"Combined val_score: {val_score:.4f}")
            
            # Report metrics for orchestrator
            print(f"\nMETRIC:val_score={val_score:.4f}")
            print(f"METRIC:detection_map={det_map:.4f}")
            print(f"METRIC:classification_map={cls_map:.4f}")
            print(f"METRIC:num_predictions={len(predictions)}")
            print(f"METRIC:model_type=yolov8x")
            print(f"METRIC:image_size=1280")
            print(f"METRIC:epochs=100")
            print(f"METRIC:batch_size={batch_size}")
            
        else:
            print("\nError: No predictions generated")
            print(f"METRIC:val_score=0.0000")
            print(f"METRIC:detection_map=0.0000")
            print(f"METRIC:classification_map=0.0000")
            print(f"METRIC:num_predictions=0")
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Report failure metrics
        print(f"METRIC:val_score=0.0000")
        print(f"METRIC:detection_map=0.0000")
        print(f"METRIC:classification_map=0.0000")
        print(f"METRIC:training_failed=1")

def main():
    """Main training pipeline."""
    print("=== YOLOv8x 1280px Training ===")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("Warning: CUDA not available, using CPU")
    
    # Train YOLOv8x at 1280px
    train_yolov8x_1280()
    
    print("\n=== YOLOv8x 1280px Training Complete ===")

if __name__ == "__main__":
    main()