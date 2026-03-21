import json
import yaml
from pathlib import Path
import torch

# Set weights_only=False for ultralytics compatibility
torch.serialization.add_safe_globals(['torch.nn.modules.container.ModuleList'])

from ultralytics import YOLO
from utils import (
    create_train_val_split, 
    setup_yolo_dataset_structure, 
    test_evaluation_pipeline,
    yolo_predictions_to_coco,
    evaluate_coco_predictions
)

def main():
    print("=== Training YOLOv8x Multiclass Baseline ===")
    
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
    
    # 3. Create YAML config for multiclass training
    config = {
        'path': str(Path.cwd() / 'datasets'),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 356,  # Number of classes (0-355)
        'names': {}
    }
    
    # Load category names from annotations
    with open('data/train/annotations.json', 'r') as f:
        coco_data = json.load(f)
    
    for cat in coco_data['categories']:
        config['names'][cat['id']] = cat['name']
    
    # Save config
    config_path = 'data_multiclass.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created config: {config_path}")
    print(f"  Classes: {config['nc']}")
    print(f"  Train path: {config['path']}/{config['train']}")
    print(f"  Val path: {config['path']}/{config['val']}")
    
    # 4. Initialize YOLOv8x model with weights_only=False
    print("\nInitializing YOLOv8x model...")
    
    # Temporarily disable weights_only for YOLO model loading
    original_weights_only = torch.serialization.DEFAULT_WEIGHTS_ONLY
    torch.serialization.DEFAULT_WEIGHTS_ONLY = False
    
    try:
        model = YOLO('yolov8x.pt')  # Load pretrained weights
    finally:
        torch.serialization.DEFAULT_WEIGHTS_ONLY = original_weights_only
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 5. Train model
    print("\nStarting training...")
    print(f"Configuration:")
    print(f"  Model: YOLOv8x")
    print(f"  Classes: {config['nc']}")
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
            name='yolov8x_multiclass_baseline',
            save_period=25,  # Save checkpoint every 25 epochs
            patience=50,  # Early stopping patience
            close_mosaic=50,  # Close mosaic augmentation after 50 epochs
            verbose=True
        )
        
        print("\nTraining completed successfully!")
        
        # 6. Load best model for evaluation
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        print(f"\nLoading best model: {best_model_path}")
        
        # Load best model with weights_only=False
        torch.serialization.DEFAULT_WEIGHTS_ONLY = False
        try:
            best_model = YOLO(best_model_path)
        finally:
            torch.serialization.DEFAULT_WEIGHTS_ONLY = original_weights_only
        
        # 7. Run inference on validation set
        print("\nRunning inference on validation set...")
        
        # Load val split info
        with open('val_split.json', 'r') as f:
            val_data = json.load(f)
        
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
        
        # Convert to COCO format
        coco_predictions = yolo_predictions_to_coco(predictions, val_image_info, score_threshold=0.01)
        
        print(f"Generated {len(coco_predictions)} predictions")
        
        # 8. Evaluate using our evaluation function
        print("\nEvaluating predictions...")
        eval_results = evaluate_coco_predictions('val_split.json', coco_predictions, verbose=True)
        
        # 9. Output metrics
        detection_mAP = eval_results['detection_mAP']
        classification_mAP = eval_results['classification_mAP']
        val_score = eval_results['val_score']
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Detection mAP@0.5: {detection_mAP:.4f}")
        print(f"Classification mAP@0.5: {classification_mAP:.4f}")
        print(f"Val Score (0.7*det + 0.3*cls): {val_score:.4f}")
        
        # Training metrics from ultralytics
        train_results = results.results_dict if hasattr(results, 'results_dict') else {}
        final_epoch = len(results.metrics['train/box_loss']) if hasattr(results, 'metrics') else 100
        
        print(f"\nTraining completed in {final_epoch} epochs")
        print(f"Best model saved to: {best_model_path}")
        
        # Output all metrics
        print(f"\nMETRIC:model=yolov8x")
        print(f"METRIC:nc={config['nc']}")
        print(f"METRIC:imgsz=1280")
        print(f"METRIC:epochs={final_epoch}")
        print(f"METRIC:detection_mAP={detection_mAP:.4f}")
        print(f"METRIC:classification_mAP={classification_mAP:.4f}")
        print(f"METRIC:val_score={val_score:.4f}")
        print(f"METRIC:predictions_count={len(coco_predictions)}")
        print(f"METRIC:val_images={len(val_image_info)}")
        print(f"METRIC:training_success=1")
        
        # Save predictions for analysis
        pred_file = 'yolov8x_baseline_predictions.json'
        with open(pred_file, 'w') as f:
            json.dump(coco_predictions, f)
        print(f"\nPredictions saved to: {pred_file}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print(f"METRIC:training_success=0")
        print(f"METRIC:error={str(e)[:100]}")
        
        # Still output basic info
        print(f"METRIC:model=yolov8x")
        print(f"METRIC:nc={config['nc']}")
        print(f"METRIC:imgsz=1280")
        print(f"METRIC:detection_mAP=0.0")
        print(f"METRIC:classification_mAP=0.0")
        print(f"METRIC:val_score=0.0")
        
        raise e
    
    print("\n=== YOLOv8x Multiclass Baseline Complete ===")

if __name__ == "__main__":
    main()