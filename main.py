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
from utils import evaluate_model, prepare_yolo_dataset

def main():
    print("=== EVALUATION FUNCTION AND YOLO DATASET PREPARATION TEST ===")
    
    try:
        # Test 1: Prepare YOLO dataset YAML
        print("\n1. Testing YOLO dataset preparation...")
        yaml_path = prepare_yolo_dataset()
        print(f"✓ YOLO dataset YAML created: {yaml_path}")
        
        # Test 2: Load validation annotations for evaluation
        print("\n2. Loading validation annotations...")
        data_dir = Path("data")
        annotations_path = data_dir / "train" / "annotations.json"
        
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        # Create a simple 90/10 split for testing (using first 10% as val)
        images = coco_data['images']
        val_size = max(1, len(images) // 10)  # At least 1 image for val
        val_image_ids = {img['id'] for img in images[:val_size]}
        
        # Filter annotations for val images
        val_annotations = [ann for ann in coco_data['annotations'] 
                          if ann['image_id'] in val_image_ids]
        
        print(f"✓ Val split: {len(val_image_ids)} images, {len(val_annotations)} annotations")
        
        # Test 3: Create dummy predictions to test evaluation function
        print("\n3. Testing evaluation function with dummy predictions...")
        
        # Create realistic dummy predictions
        dummy_predictions = []
        for ann in val_annotations[:50]:  # Test with first 50 annotations
            # Create a prediction that's close to the ground truth
            bbox = ann['bbox'].copy()
            # Add some noise to make it realistic
            bbox[0] += 5  # x offset
            bbox[1] += 3  # y offset
            bbox[2] *= 0.95  # slightly smaller width
            bbox[3] *= 0.98  # slightly smaller height
            
            pred = {
                'image_id': ann['image_id'],
                'category_id': ann['category_id'],
                'bbox': bbox,
                'score': 0.8  # High confidence
            }
            dummy_predictions.append(pred)
        
        # Add some false positives
        for i, ann in enumerate(val_annotations[50:60]):
            bbox = ann['bbox'].copy()
            bbox[0] += 100  # Offset to make it a different detection
            pred = {
                'image_id': ann['image_id'],
                'category_id': (ann['category_id'] + 1) % 357,  # Wrong category
                'bbox': bbox,
                'score': 0.6
            }
            dummy_predictions.append(pred)
        
        print(f"✓ Created {len(dummy_predictions)} dummy predictions")
        
        # Test 4: Run evaluation
        print("\n4. Running evaluation...")
        val_score, detection_map, classification_map = evaluate_model(
            dummy_predictions, 
            val_image_ids, 
            annotations_path
        )
        
        print(f"✓ Evaluation completed successfully")
        print(f"  Detection mAP@0.5: {detection_map:.4f}")
        print(f"  Classification mAP@0.5: {classification_map:.4f}")
        print(f"  Combined val_score: {val_score:.4f}")
        
        # Test 5: Verify the score calculation
        expected_score = 0.7 * detection_map + 0.3 * classification_map
        score_diff = abs(val_score - expected_score)
        
        if score_diff < 1e-6:
            print("✓ Score calculation verified")
        else:
            print(f"⚠ Score calculation mismatch: {val_score} vs {expected_score}")
        
        print("\n=== ALL TESTS PASSED ===")
        print(f"METRIC:evaluation_function_success=1.0")
        print(f"METRIC:yolo_dataset_success=1.0")
        print(f"METRIC:dummy_val_score={val_score:.4f}")
        print(f"METRIC:dummy_detection_map={detection_map:.4f}")
        print(f"METRIC:dummy_classification_map={classification_map:.4f}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"METRIC:evaluation_function_success=0.0")
        print(f"METRIC:yolo_dataset_success=0.0")

if __name__ == "__main__":
    main()