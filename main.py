import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from utils import evaluate_predictions, convert_coco_to_yolo

def test_evaluation_function():
    """Test the evaluation function with dummy predictions"""
    print("=== Testing Evaluation Function ===")
    
    # Load val split to get ground truth
    val_split_path = Path("data/val_split.json")
    if not val_split_path.exists():
        print(f"❌ Val split not found at {val_split_path}")
        return False
    
    with open(val_split_path, 'r') as f:
        val_split = json.load(f)
    
    print(f"Val split: {len(val_split['images'])} images, {len(val_split['annotations'])} annotations")
    
    # Create dummy predictions - mix of correct and incorrect
    dummy_predictions = []
    
    for i, ann in enumerate(val_split['annotations'][:50]):  # Test with first 50 annotations
        # Create a prediction that's close to the ground truth
        bbox = ann['bbox'].copy()
        
        # Add some noise to bbox
        bbox[0] += random.uniform(-5, 5)  # x offset
        bbox[1] += random.uniform(-5, 5)  # y offset
        bbox[2] *= random.uniform(0.9, 1.1)  # width scale
        bbox[3] *= random.uniform(0.9, 1.1)  # height scale
        
        # Sometimes use correct category, sometimes wrong
        if i % 3 == 0:  # 1/3 correct classifications
            category_id = ann['category_id']
        else:  # 2/3 wrong classifications
            category_id = random.randint(0, 356)
        
        dummy_predictions.append({
            'image_id': ann['image_id'],
            'category_id': category_id,
            'bbox': bbox,
            'score': random.uniform(0.5, 0.95)
        })
    
    print(f"Created {len(dummy_predictions)} dummy predictions")
    
    # Test evaluation
    try:
        val_score, det_map, cls_map = evaluate_predictions(dummy_predictions, val_split)
        print(f"✓ Evaluation successful:")
        print(f"  Detection mAP@0.5: {det_map:.4f}")
        print(f"  Classification mAP@0.5: {cls_map:.4f}")
        print(f"  Combined val_score: {val_score:.4f}")
        return True
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_yolo_conversion():
    """Test COCO to YOLO format conversion"""
    print("\n=== Testing YOLO Conversion ===")
    
    # Check if splits exist
    train_split_path = Path("data/train_split.json")
    val_split_path = Path("data/val_split.json")
    
    if not train_split_path.exists() or not val_split_path.exists():
        print("❌ Train/val splits not found")
        return False
    
    # Convert to YOLO format
    try:
        train_success = convert_coco_to_yolo(
            coco_json_path=train_split_path,
            images_dir=Path("data/train/images"),
            output_dir=Path("data/yolo_train"),
            split_name="train"
        )
        
        val_success = convert_coco_to_yolo(
            coco_json_path=val_split_path,
            images_dir=Path("data/train/images"),
            output_dir=Path("data/yolo_val"),
            split_name="val"
        )
        
        if train_success and val_success:
            print("✓ YOLO conversion successful")
            
            # Check output structure
            train_dir = Path("data/yolo_train")
            val_dir = Path("data/yolo_val")
            data_yaml = Path("data/data.yaml")
            
            print(f"  Train images: {len(list(train_dir.glob('*.jpg')))}")
            print(f"  Train labels: {len(list(train_dir.glob('*.txt')))}")
            print(f"  Val images: {len(list(val_dir.glob('*.jpg')))}")
            print(f"  Val labels: {len(list(val_dir.glob('*.txt')))}")
            print(f"  Data YAML exists: {data_yaml.exists()}")
            
            return True
        else:
            print("❌ YOLO conversion failed")
            return False
            
    except Exception as e:
        print(f"❌ YOLO conversion error: {e}")
        import traceback
        traceback.print_exc()
        return False

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
    
    # Print category distribution stats
    train_cats = Counter(ann['category_id'] for ann in train_annotations)
    val_cats = Counter(ann['category_id'] for ann in val_annotations)
    
    print(f"\nCategory distribution:")
    print(f"Train: {len(train_cats)} unique categories")
    print(f"Val: {len(val_cats)} unique categories")
    
    # Categories only in train or val
    train_only = set(train_cats.keys()) - set(val_cats.keys())
    val_only = set(val_cats.keys()) - set(train_cats.keys())
    
    if train_only:
        print(f"Categories only in train: {len(train_only)}")
    if val_only:
        print(f"Categories only in val: {len(val_only)}")
    
    return True

def check_environment():
    """Check PyTorch, CUDA, and required packages"""
    print("\n=== Environment Check ===")
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        torch_ok = True
    except ImportError:
        print("❌ PyTorch not available")
        torch_ok = False
    
    # Check ultralytics
    try:
        import ultralytics
        from ultralytics import YOLO
        print(f"✓ Ultralytics version: {ultralytics.__version__}")
        ultralytics_ok = True
    except ImportError:
        print("❌ Ultralytics not available")
        ultralytics_ok = False
    
    # Check pycocotools
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        print("✓ pycocotools available")
        coco_ok = True
    except ImportError:
        print("❌ pycocotools not available")
        coco_ok = False
    
    # Check other packages
    other_packages = ['numpy', 'scipy', 'scikit-learn', 'timm']
    other_ok = True
    
    for pkg in other_packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg} available")
        except ImportError:
            print(f"❌ {pkg} not available")
            other_ok = False
    
    all_ok = torch_ok and ultralytics_ok and coco_ok and other_ok
    return all_ok

def validate_splits():
    """Validate the created splits"""
    print("\n=== Split Validation ===")
    
    # Check if split files exist
    train_split_path = Path("data/train_split.json")
    val_split_path = Path("data/val_split.json")
    
    if not train_split_path.exists():
        print(f"❌ Train split not found at {train_split_path}")
        return False
    
    if not val_split_path.exists():
        print(f"❌ Val split not found at {val_split_path}")
        return False
    
    # Load splits
    with open(train_split_path, 'r') as f:
        train_split = json.load(f)
    
    with open(val_split_path, 'r') as f:
        val_split = json.load(f)
    
    # Check COCO format
    required_keys = ['images', 'annotations', 'categories']
    for split_name, split_data in [("train", train_split), ("val", val_split)]:
        for key in required_keys:
            if key not in split_data:
                print(f"❌ {split_name} split missing key: {key}")
                return False
    
    # Check for overlap
    train_image_ids = set(img['id'] for img in train_split['images'])
    val_image_ids = set(img['id'] for img in val_split['images'])
    
    overlap = train_image_ids.intersection(val_image_ids)
    if overlap:
        print(f"❌ Found {len(overlap)} overlapping image IDs")
        return False
    
    print(f"✓ Train: {len(train_split['images'])} images, {len(train_split['annotations'])} annotations")
    print(f"✓ Val: {len(val_split['images'])} images, {len(val_split['annotations'])} annotations")
    print("✓ No overlap between splits")
    print("✓ Valid COCO format")
    
    return True

def main():
    """Main experiment: Build evaluation function and YOLO dataset conversion"""
    print("=== Step 4: Build Evaluation Function and YOLO Dataset Conversion ===")
    
    # Ensure splits exist
    if not Path("data/train_split.json").exists():
        print("Creating train/val splits first...")
        splits_created = create_train_val_splits()
        if not splits_created:
            print("❌ Failed to create splits")
            return
    
    # Test evaluation function
    eval_success = test_evaluation_function()
    
    # Test YOLO conversion
    yolo_success = test_yolo_conversion()
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Evaluation function: {'✓' if eval_success else '❌'}")
    print(f"YOLO conversion: {'✓' if yolo_success else '❌'}")
    
    # Metrics for tracking
    print(f"METRIC:eval_function_works={1 if eval_success else 0}")
    print(f"METRIC:yolo_conversion_works={1 if yolo_success else 0}")
    
    if eval_success and yolo_success:
        print("\n🎉 Ready for YOLO training experiments!")
        
        # Check YOLO dataset structure
        train_images = len(list(Path("data/yolo_train").glob("*.jpg")))
        train_labels = len(list(Path("data/yolo_train").glob("*.txt")))
        val_images = len(list(Path("data/yolo_val").glob("*.jpg")))
        val_labels = len(list(Path("data/yolo_val").glob("*.txt")))
        
        print(f"METRIC:train_images_yolo={train_images}")
        print(f"METRIC:train_labels_yolo={train_labels}")
        print(f"METRIC:val_images_yolo={val_images}")
        print(f"METRIC:val_labels_yolo={val_labels}")
    else:
        print("\n⚠ Issues found - check logs above")

if __name__ == "__main__":
    main()