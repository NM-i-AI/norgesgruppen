import torch

# Monkey-patch torch.load to fix ultralytics compatibility with PyTorch 2.6
# PyTorch 2.6 changed default weights_only=True for security, but ultralytics expects False
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Now import ultralytics after patching
from ultralytics import YOLO
import json
import os
from pathlib import Path
from utils import evaluate_model, create_train_val_split, save_coco_split

def convert_coco_to_yolo_labels(coco_data, output_dir, image_ids=None):
    """
    Convert COCO annotations to YOLO label format.
    
    Args:
        coco_data: COCO dataset dict
        output_dir: Directory to save YOLO label files
        image_ids: Set of image IDs to process (if None, process all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create image_id to filename mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if image_ids is None or img_id in image_ids:
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
    
    # Convert each image's annotations
    for img_id, annotations in annotations_by_image.items():
        if img_id not in image_info:
            continue
            
        img_info = image_info[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Create label filename (same as image but .txt)
        img_filename = img_info['file_name']
        label_filename = Path(img_filename).stem + '.txt'
        label_path = output_dir / label_filename
        
        # Convert annotations to YOLO format
        yolo_lines = []
        for ann in annotations:
            # COCO bbox format: [x, y, width, height] (absolute pixels)
            x, y, w, h = ann['bbox']
            
            # Convert to YOLO format: [class_id, x_center, y_center, width, height] (normalized)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            norm_width = w / img_width
            norm_height = h / img_height
            
            # YOLO expects 0-indexed classes
            class_id = ann['category_id']
            
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
            yolo_lines.append(yolo_line)
        
        # Write label file
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
    
    print(f"Converted {len(annotations_by_image)} images to YOLO format in {output_dir}")

def create_yolo_dataset_structure():
    """
    Create proper YOLO dataset structure with train/val splits.
    
    Returns:
        str: Path to dataset YAML file
    """
    data_dir = Path("data")
    annotations_path = data_dir / "train" / "annotations.json"
    images_dir = data_dir / "train" / "images"
    
    # Load COCO data
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create train/val split
    train_image_ids, val_image_ids = create_train_val_split(annotations_path, val_ratio=0.1, seed=42)
    
    print(f"Split: {len(train_image_ids)} train, {len(val_image_ids)} val images")
    
    # Create YOLO dataset directories
    yolo_dir = Path("yolo_dataset")
    train_images_dir = yolo_dir / "train" / "images"
    train_labels_dir = yolo_dir / "train" / "labels"
    val_images_dir = yolo_dir / "val" / "images"
    val_labels_dir = yolo_dir / "val" / "labels"
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Convert annotations to YOLO format
    convert_coco_to_yolo_labels(coco_data, train_labels_dir, train_image_ids)
    convert_coco_to_yolo_labels(coco_data, val_labels_dir, val_image_ids)
    
    # Create symlinks to images (or copy if symlinks not supported)
    def link_images(image_ids, target_dir):
        for img_id in image_ids:
            # Find image info
            img_info = next(img for img in coco_data['images'] if img['id'] == img_id)
            src_path = images_dir / img_info['file_name']
            dst_path = target_dir / img_info['file_name']
            
            if src_path.exists() and not dst_path.exists():
                try:
                    os.symlink(src_path.absolute(), dst_path)
                except OSError:
                    # Fallback to copy if symlinks not supported
                    import shutil
                    shutil.copy2(src_path, dst_path)
    
    link_images(train_image_ids, train_images_dir)
    link_images(val_image_ids, val_images_dir)
    
    print(f"Created YOLO dataset structure in {yolo_dir}")
    
    # Create dataset YAML
    yaml_config = {
        'path': str(yolo_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(coco_data['categories']),
        'names': {cat['id']: cat['name'] for cat in coco_data['categories']}
    }
    
    yaml_path = "grocery_yolo.yaml"
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    return yaml_path, train_image_ids, val_image_ids

def main():
    print("=== YOLO DATASET PREPARATION AND YOLOv8m TRAINING ===")
    
    try:
        # Step 1: Create YOLO dataset structure
        print("\n1. Creating YOLO dataset structure...")
        yaml_path, train_image_ids, val_image_ids = create_yolo_dataset_structure()
        print(f"✓ Dataset YAML created: {yaml_path}")
        
        # Step 2: Initialize and train YOLOv8m model
        print("\n2. Initializing YOLOv8m model...")
        model = YOLO('yolov8m.pt')  # Use medium model instead of x
        print("✓ YOLOv8m model loaded")
        
        # Step 3: Train the model
        print("\n3. Starting training...")
        print(f"Training config: YOLOv8m, 640px, 30 epochs, batch=-1 (auto)")
        
        results = model.train(
            data=yaml_path,
            epochs=30,  # Reduced from 50 for faster iteration
            imgsz=640,  # Reduced from 1280 to avoid OOM
            batch=-1,   # Auto-determine batch size
            device=0,   # Use first GPU
            project='runs/detect',
            name='yolov8m_nc356_640px_30ep',
            save=True,
            save_period=10,  # Save every 10 epochs
            val=True,
            plots=True,
            verbose=True
        )
        
        print("✓ Training completed successfully")
        
        # Step 4: Load best model and run inference on validation set
        print("\n4. Running validation inference...")
        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
        best_model = YOLO(str(best_model_path))
        
        # Run inference on validation images
        val_images_dir = Path("yolo_dataset") / "val" / "images"
        val_image_paths = list(val_images_dir.glob("*.jpg"))
        
        print(f"Running inference on {len(val_image_paths)} validation images...")
        
        predictions = []
        for img_path in val_image_paths:
            # Extract image_id from filename
            img_id = int(img_path.stem.split('_')[-1])  # Assumes format like img_00001.jpg
            
            # Run inference
            results_list = best_model(str(img_path), verbose=False)
            
            # Convert results to COCO format
            for result in results_list:
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        # Extract box data
                        xyxy = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy())
                        
                        # Convert xyxy to xywh (COCO format)
                        x1, y1, x2, y2 = xyxy
                        x, y, w, h = x1, y1, x2 - x1, y2 - y1
                        
                        pred = {
                            'image_id': img_id,
                            'category_id': cls,
                            'bbox': [float(x), float(y), float(w), float(h)],
                            'score': conf
                        }
                        predictions.append(pred)
        
        print(f"✓ Generated {len(predictions)} predictions")
        
        # Step 5: Evaluate using competition metric
        print("\n5. Evaluating model performance...")
        annotations_path = Path("data") / "train" / "annotations.json"
        
        val_score, detection_map, classification_map = evaluate_model(
            predictions, val_image_ids, annotations_path
        )
        
        print(f"\n=== TRAINING RESULTS ===\n")
        print(f"Model: YOLOv8m")
        print(f"Resolution: 640px")
        print(f"Epochs: 30")
        print(f"Validation images: {len(val_image_ids)}")
        print(f"Predictions generated: {len(predictions)}")
        print(f"\nPerformance Metrics:")
        print(f"  Detection mAP@0.5: {detection_map:.4f}")
        print(f"  Classification mAP@0.5: {classification_map:.4f}")
        print(f"  Combined val_score: {val_score:.4f}")
        print(f"\nModel saved to: {best_model_path}")
        
        # Output metrics for tracking
        print(f"\nMETRIC:val_score={val_score:.4f}")
        print(f"METRIC:detection_map={detection_map:.4f}")
        print(f"METRIC:classification_map={classification_map:.4f}")
        print(f"METRIC:training_success=1.0")
        print(f"METRIC:num_predictions={len(predictions)}")
        print(f"METRIC:model_size=medium")
        print(f"METRIC:resolution=640")
        print(f"METRIC:epochs=30")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"METRIC:val_score=0.0")
        print(f"METRIC:training_success=0.0")
        print(f"METRIC:training_failed=1.0")

if __name__ == "__main__":
    main()