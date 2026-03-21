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

def main():
    print("=== PYTORCH/ULTRALYTICS COMPATIBILITY FIX TEST ===")
    
    try:
        # Test 1: Load a pretrained YOLO model
        print("\n1. Testing YOLO model loading...")
        model = YOLO('yolov8n.pt')
        print("✓ YOLOv8n model loaded successfully")
        
        # Test 2: Check model info
        print("\n2. Model information:")
        info = model.info(verbose=False)
        print(f"✓ Model info retrieved: {info}")
        
        # Test 3: Load training data to verify data path
        print("\n3. Checking training data...")
        data_dir = Path("data")
        annotations_path = data_dir / "train" / "annotations.json"
        
        if annotations_path.exists():
            with open(annotations_path, 'r') as f:
                coco_data = json.load(f)
            
            num_images = len(coco_data['images'])
            num_annotations = len(coco_data['annotations'])
            num_categories = len(coco_data['categories'])
            
            print(f"✓ Training data found: {num_images} images, {num_annotations} annotations, {num_categories} categories")
            
            # Test 4: Try a very short training run (1 epoch) to verify everything works
            print("\n4. Testing short training run (1 epoch)...")
            
            # Create a minimal dataset config for YOLO
            train_images_dir = data_dir / "train" / "images"
            
            # Convert COCO to YOLO format for this test (minimal)
            yolo_data_config = {
                'path': str(data_dir.absolute()),
                'train': str(train_images_dir.relative_to(data_dir)),
                'val': str(train_images_dir.relative_to(data_dir)),  # Use same for quick test
                'nc': num_categories,
                'names': {cat['id']: cat['name'] for cat in coco_data['categories']}
            }
            
            # Save temporary config
            config_path = Path("temp_test_config.yaml")
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(yolo_data_config, f)
            
            try:
                # Try training for 1 epoch with minimal settings
                results = model.train(
                    data=str(config_path),
                    epochs=1,
                    imgsz=640,
                    batch=1,  # Small batch to avoid memory issues
                    device=0 if torch.cuda.is_available() else 'cpu',
                    verbose=True,
                    save=False,  # Don't save weights
                    plots=False,  # Don't generate plots
                    val=False   # Skip validation for speed
                )
                print("✓ Training completed successfully for 1 epoch")
                
                # Clean up
                if config_path.exists():
                    config_path.unlink()
                
                print("\n=== COMPATIBILITY FIX SUCCESSFUL ===")
                print("METRIC:test_training_success=1.0")
                print("METRIC:model_loading_success=1.0")
                
            except Exception as train_error:
                print(f"⚠ Training test failed: {train_error}")
                print("Model loading works but training needs debugging")
                print("METRIC:test_training_success=0.0")
                print("METRIC:model_loading_success=1.0")
                
                # Clean up
                if config_path.exists():
                    config_path.unlink()
        
        else:
            print("⚠ Training data not found, but model loading works")
            print("METRIC:test_training_success=0.0")
            print("METRIC:model_loading_success=1.0")
            
    except Exception as e:
        print(f"❌ YOLO model loading failed: {e}")
        print("Compatibility fix did not resolve the issue")
        print("METRIC:test_training_success=0.0")
        print("METRIC:model_loading_success=0.0")
        
        # Try alternative fix - check if it's a different issue
        print("\nTrying alternative diagnostics...")
        try:
            import ultralytics
            print(f"Ultralytics version: {ultralytics.__version__}")
            print(f"PyTorch version: {torch.__version__}")
            
            # Check if the issue is with specific model files
            print("Checking torch.hub cache...")
            hub_dir = Path.home() / '.cache' / 'torch' / 'hub'
            if hub_dir.exists():
                print(f"Hub cache exists: {hub_dir}")
                cache_files = list(hub_dir.glob('**/*.pt'))
                print(f"Found {len(cache_files)} cached .pt files")
            
        except Exception as diag_error:
            print(f"Diagnostic error: {diag_error}")

if __name__ == "__main__":
    main()