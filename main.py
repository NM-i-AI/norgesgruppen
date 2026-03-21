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
from pathlib import Path
from utils import evaluate_model, prepare_yolo_dataset, create_train_val_split, save_coco_split

def main():
    print("=== YOLOv8x SINGLE-CLASS DETECTOR TRAINING ===")
    
    try:
        # Step 1: Create train/val split
        print("\n1. Creating train/val split...")
        data_dir = Path("data")
        annotations_path = data_dir / "train" / "annotations.json"
        
        # Load original annotations
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        # Create 90/10 split
        train_image_ids, val_image_ids = create_train_val_split(
            annotations_path, val_ratio=0.1, seed=42
        )
        
        print(f"✓ Split created: {len(train_image_ids)} train, {len(val_image_ids)} val images")
        
        # Step 2: Create single-class annotations
        print("\n2. Converting to single-class annotations...")
        
        # Convert all annotations to class 0 (single product class)
        single_class_coco = coco_data.copy()
        single_class_coco['categories'] = [{'id': 0, 'name': 'product'}]
        
        # Map all annotations to class 0
        for ann in single_class_coco['annotations']:
            ann['category_id'] = 0
        
        print(f"✓ Converted {len(single_class_coco['annotations'])} annotations to single class")
        
        # Save splits with single-class annotations
        train_count, train_ann_count = save_coco_split(
            single_class_coco, train_image_ids, "train_split_nc1.json"
        )
        val_count, val_ann_count = save_coco_split(
            single_class_coco, val_image_ids, "val_split_nc1.json"
        )
        
        print(f"✓ Train split: {train_count} images, {train_ann_count} annotations")
        print(f"✓ Val split: {val_count} images, {val_ann_count} annotations")
        
        # Step 3: Prepare YOLO dataset YAML for single class
        print("\n3. Preparing single-class YOLO dataset configuration...")
        
        # Create single-class YOLO config
        yolo_config = {
            'path': str(data_dir.absolute()),
            'train': 'train/images',
            'val': 'train/images',
            'nc': 1,
            'names': {0: 'product'}
        }
        
        yaml_path = Path("grocery_dataset_nc1.yaml")
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(yolo_config, f, default_flow_style=False)
        
        print(f"✓ YOLO dataset config: {yaml_path}")
        print(f"✓ Number of classes: {yolo_config['nc']} (single product class)")
        
        # Step 4: Initialize YOLOv8x model
        print("\n4. Initializing YOLOv8x model...")
        model = YOLO('yolov8x.pt')  # This will download pretrained weights
        print(f"✓ YOLOv8x model loaded")
        
        # Step 5: Train the model
        print("\n5. Starting single-class training...")
        print(f"Training parameters:")
        print(f"  - Model: YOLOv8x")
        print(f"  - Classes: 1 (nc=1, single product class)")
        print(f"  - Image size: 1280")
        print(f"  - Epochs: 50")
        print(f"  - Close mosaic: 10")
        print(f"  - Batch: auto (should be larger than nc=356 case)")
        
        # Train with specified parameters
        results = model.train(
            data=str(yaml_path),
            epochs=50,
            imgsz=1280,
            batch=-1,  # auto batch size (should be larger for nc=1)
            close_mosaic=10,
            device=0,  # Use first GPU
            project="runs/detect",
            name="yolov8x_nc1",
            save=True,
            save_period=10,  # Save every 10 epochs
            val=True,
            plots=True,
            verbose=True
        )
        
        print(f"✓ Training completed")
        
        # Step 6: Load best model and run inference on validation set
        print("\n6. Evaluating on validation split...")
        
        # Load the best model from training
        best_model_path = Path("runs/detect/yolov8x_nc1/weights/best.pt")
        if not best_model_path.exists():
            # Fallback to last.pt if best.pt doesn't exist
            best_model_path = Path("runs/detect/yolov8x_nc1/weights/last.pt")
        
        eval_model = YOLO(str(best_model_path))
        print(f"✓ Loaded model: {best_model_path}")
        
        # Run inference on validation images
        val_predictions = []
        images_dir = data_dir / "train" / "images"
        
        # Load val split to get image filenames
        with open("val_split_nc1.json", 'r') as f:
            val_data = json.load(f)
        
        val_image_files = {img['id']: img['file_name'] for img in val_data['images']}
        
        print(f"Running inference on {len(val_image_files)} validation images...")
        
        for image_id, filename in val_image_files.items():
            image_path = images_dir / filename
            
            # Run inference
            results = eval_model(str(image_path), conf=0.01, iou=0.7, verbose=False)
            
            # Convert results to COCO format
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        # Get box coordinates in COCO format [x, y, width, height]
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        
                        # Get class and confidence (should all be class 0)
                        cls = int(boxes.cls[i].cpu().numpy())
                        conf = float(boxes.conf[i].cpu().numpy())
                        
                        pred = {
                            'image_id': image_id,
                            'category_id': cls,  # Should be 0 for all predictions
                            'bbox': bbox,
                            'score': conf
                        }
                        val_predictions.append(pred)
        
        print(f"✓ Generated {len(val_predictions)} predictions")
        
        # Step 7: Evaluate using our competition metric
        print("\n7. Computing competition metric...")
        
        # For single-class evaluation, we need to use the original multiclass annotations
        # to compute detection mAP properly
        val_score, detection_map, classification_map = evaluate_model(
            val_predictions, 
            val_image_ids, 
            annotations_path  # Use original multiclass annotations
        )
        
        print(f"\n=== SINGLE-CLASS DETECTOR RESULTS ===")
        print(f"Detection mAP@0.5: {detection_map:.4f}")
        print(f"Classification mAP@0.5: {classification_map:.4f}")
        print(f"Combined val_score: {val_score:.4f}")
        print(f"Formula: 0.7 × {detection_map:.4f} + 0.3 × {classification_map:.4f} = {val_score:.4f}")
        
        # Note about single-class performance
        print(f"\nNote: Single-class detector (nc=1) should excel at detection but")
        print(f"will score 0 on classification since all predictions are category_id=0.")
        print(f"Maximum possible val_score for nc=1 model: 0.7 × detection_mAP = {0.7 * detection_map:.4f}")
        
        # Output metrics for tracking
        print(f"\nMETRIC:val_score={val_score:.6f}")
        print(f"METRIC:detection_map={detection_map:.6f}")
        print(f"METRIC:classification_map={classification_map:.6f}")
        print(f"METRIC:num_predictions={len(val_predictions)}")
        print(f"METRIC:training_epochs=50")
        print(f"METRIC:model_size=yolov8x")
        print(f"METRIC:image_size=1280")
        print(f"METRIC:num_classes=1")
        print(f"METRIC:max_possible_score={0.7 * detection_map:.6f}")
        
        # Save predictions for analysis
        with open("nc1_predictions.json", 'w') as f:
            json.dump(val_predictions, f)
        print(f"✓ Predictions saved to nc1_predictions.json")
        
        print(f"\n=== SINGLE-CLASS TRAINING COMPLETE ===")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"METRIC:val_score=0.0")
        print(f"METRIC:training_failed=1.0")

if __name__ == "__main__":
    main()