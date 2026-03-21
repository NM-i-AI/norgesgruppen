import json
import yaml
from pathlib import Path
import torch

# Fix for PyTorch 2.6 weights_only issue with ultralytics models
# Monkey patch torch.load to use weights_only=False for ultralytics compatibility
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
    # Force weights_only=False for ultralytics compatibility
    kwargs['weights_only'] = False
    return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)

# Apply the patch
torch.load = patched_torch_load

# Import ultralytics after patching torch.load
from ultralytics import YOLO
from utils import (
    create_train_val_split, 
    setup_yolo_dataset_structure, 
    test_evaluation_pipeline,
    yolo_predictions_to_coco,
    evaluate_coco_predictions
)

def main():
    print("=== Training YOLOv8x Single-Class Detector ===")
    
    # 1. Verify splits exist
    if not Path('train_split.json').exists() or not Path('val_split.json').exists():
        print("Creating train/val splits...")
        create_train_val_split(
            annotations_path='data/train/annotations.json',
            train_output='train_split.json',
            val_output='val_split.json',
            val_ratio=0.1,
            seed=42
        )
    
    # 2. Setup YOLO dataset structure if needed
    if not Path('datasets/train/images').exists():
        print("Setting up YOLO dataset structure...")
        setup_yolo_dataset_structure()
    
    # 3. Create single-class YOLO labels (convert all categories to class 0)
    print("Creating single-class YOLO labels...")
    
    # Load train and val splits
    with open('train_split.json', 'r') as f:
        train_data = json.load(f)
    with open('val_split.json', 'r') as f:
        val_data = json.load(f)
    
    # Create single-class labels for train
    train_labels_dir = Path('datasets/train/labels')
    for img_info in train_data['images']:
        label_file = train_labels_dir / (Path(img_info['file_name']).stem + '.txt')
        if label_file.exists():
            # Read existing labels and convert all classes to 0
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            with open(label_file, 'w') as f:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Change class to 0, keep bbox coordinates
                        f.write(f"0 {' '.join(parts[1:])}\n")
    
    # Create single-class labels for val
    val_labels_dir = Path('datasets/val/labels')
    for img_info in val_data['images']:
        label_file = val_labels_dir / (Path(img_info['file_name']).stem + '.txt')
        if label_file.exists():
            # Read existing labels and convert all classes to 0
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            with open(label_file, 'w') as f:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Change class to 0, keep bbox coordinates
                        f.write(f"0 {' '.join(parts[1:])}\n")
    
    print("Single-class labels created")
    
    # 4. Create YAML config for single-class training
    config = {
        'path': str(Path.cwd() / 'datasets'),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,  # Single class
        'names': {0: 'product'}
    }
    
    # Save config
    config_path = 'data_single_class.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created config: {config_path}")
    print(f"  Classes: {config['nc']} (single class)")
    print(f"  Train path: {config['path']}/{config['train']}")
    print(f"  Val path: {config['path']}/{config['val']}")
    
    # 5. Initialize YOLOv8x model
    print("\nInitializing YOLOv8x model...")
    model = YOLO('yolov8x.pt')  # Load pretrained weights
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 6. Train model
    print("\nStarting training...")
    print(f"Configuration:")
    print(f"  Model: YOLOv8x")
    print(f"  Classes: {config['nc']} (single class detector)")
    print(f"  Image size: 1280")
    print(f"  Epochs: 100")
    print(f"  Device: {device}")
    
    try:
        results = model.train(
            data=config_path,
            epochs=100,
            imgsz=1280,
            batch=4,  # Conservative batch size for 1280 resolution
            device=device,
            project='runs/train',
            name='yolov8x_single_class',
            save_period=25,  # Save checkpoint every 25 epochs
            patience=50,  # Early stopping patience
            close_mosaic=50,  # Close mosaic augmentation after 50 epochs
            verbose=True
        )
        
        print("\nTraining completed successfully!")
        
        # 7. Load best model for evaluation
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        print(f"\nLoading best model: {best_model_path}")
        
        best_model = YOLO(best_model_path)
        
        # 8. Run inference on validation set
        print("\nRunning inference on validation set...")
        
        # Load val split info
        val_image_info = []
        val_image_paths = []
        
        for img_info in val_data['images']:
            val_image_info.append({
                'id': img_info['id'],
                'width': img_info['width'],
                'height': img_info['height']
            })
            val_image_paths.append(f"datasets/val/images/{img_info['file_name']}")
        
        # Run inference
        predictions = best_model.predict(
            val_image_paths,
            imgsz=1280,
            conf=0.01,  # Low confidence threshold to maximize recall
            iou=0.7,    # NMS IoU threshold
            verbose=False
        )
        
        # Convert to COCO format (all predictions will have category_id=0)
        coco_predictions = yolo_predictions_to_coco(predictions, val_image_info, score_threshold=0.01)
        
        print(f"Generated {len(coco_predictions)} predictions")
        
        # 9. Evaluate using our evaluation function
        print("\nEvaluating predictions...")
        eval_results = evaluate_coco_predictions('val_split.json', coco_predictions, verbose=True)
        
        # 10. Output metrics
        detection_mAP = eval_results['detection_mAP']
        classification_mAP = eval_results['classification_mAP']
        val_score = eval_results['val_score']
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Detection mAP@0.5: {detection_mAP:.4f}")
        print(f"Classification mAP@0.5: {classification_mAP:.4f} (expected 0.0 for single-class)")
        print(f"Val Score (0.7*det + 0.3*cls): {val_score:.4f}")
        
        # Training metrics from ultralytics
        final_epoch = len(results.metrics['train/box_loss']) if hasattr(results, 'metrics') else 100
        
        print(f"\nTraining completed in {final_epoch} epochs")
        print(f"Best model saved to: {best_model_path}")
        
        # Output all metrics
        print(f"\nMETRIC:model=yolov8x")
        print(f"METRIC:nc=1")
        print(f"METRIC:imgsz=1280")
        print(f"METRIC:epochs={final_epoch}")
        print(f"METRIC:detection_mAP={detection_mAP:.4f}")
        print(f"METRIC:classification_mAP={classification_mAP:.4f}")
        print(f"METRIC:val_score={val_score:.4f}")
        print(f"METRIC:predictions_count={len(coco_predictions)}")
        print(f"METRIC:val_images={len(val_image_info)}")
        print(f"METRIC:training_success=1")
        
        # Save predictions for analysis
        pred_file = 'yolov8x_single_class_predictions.json'
        with open(pred_file, 'w') as f:
            json.dump(coco_predictions, f)
        print(f"\nPredictions saved to: {pred_file}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print(f"METRIC:training_success=0")
        print(f"METRIC:error={str(e)[:100]}")
        
        # Still output basic info
        print(f"METRIC:model=yolov8x")
        print(f"METRIC:nc=1")
        print(f"METRIC:imgsz=1280")
        print(f"METRIC:detection_mAP=0.0")
        print(f"METRIC:classification_mAP=0.0")
        print(f"METRIC:val_score=0.0")
        
        raise e
    
    print("\n=== YOLOv8x Single-Class Detector Complete ===")

if __name__ == "__main__":
    main()