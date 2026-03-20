import json
import os
from pathlib import Path
from collections import Counter, defaultdict

def validate_splits():
    print("=== Split Validation ===\n")
    
    # Load original annotations
    annotations_path = Path("data/train/annotations.json")
    if not annotations_path.exists():
        print(f"ERROR: Original annotations not found at {annotations_path}")
        return False
    
    with open(annotations_path, 'r') as f:
        original_data = json.load(f)
    
    print(f"Original dataset: {len(original_data['images'])} images, {len(original_data['annotations'])} annotations")
    
    # Load splits
    train_split_path = Path("data/train_split.json")
    val_split_path = Path("data/val_split.json")
    
    if not train_split_path.exists():
        print(f"ERROR: Train split not found at {train_split_path}")
        return False
    
    if not val_split_path.exists():
        print(f"ERROR: Val split not found at {val_split_path}")
        return False
    
    with open(train_split_path, 'r') as f:
        train_split = json.load(f)
    
    with open(val_split_path, 'r') as f:
        val_split = json.load(f)
    
    print(f"Train split: {len(train_split['images'])} images, {len(train_split['annotations'])} annotations")
    print(f"Val split: {len(val_split['images'])} images, {len(val_split['annotations'])} annotations")
    
    # Validate COCO format
    required_keys = ['images', 'annotations', 'categories']
    for split_name, split_data in [("train", train_split), ("val", val_split)]:
        for key in required_keys:
            if key not in split_data:
                print(f"ERROR: {split_name} split missing required key: {key}")
                return False
        print(f"✓ {split_name} split has valid COCO format")
    
    # Check for overlap between train and val image IDs
    train_image_ids = set(img['id'] for img in train_split['images'])
    val_image_ids = set(img['id'] for img in val_split['images'])
    
    overlap = train_image_ids.intersection(val_image_ids)
    if overlap:
        print(f"ERROR: Found {len(overlap)} overlapping image IDs between train and val: {list(overlap)[:5]}...")
        return False
    else:
        print("✓ No overlap between train and val image IDs")
    
    # Check total counts match original
    total_images = len(train_split['images']) + len(val_split['images'])
    total_annotations = len(train_split['annotations']) + len(val_split['annotations'])
    
    if total_images != len(original_data['images']):
        print(f"ERROR: Image count mismatch. Original: {len(original_data['images'])}, Split total: {total_images}")
        return False
    
    if total_annotations != len(original_data['annotations']):
        print(f"ERROR: Annotation count mismatch. Original: {len(original_data['annotations'])}, Split total: {total_annotations}")
        return False
    
    print("✓ Total counts match original dataset")
    
    # Check split ratio
    val_ratio = len(val_split['images']) / len(original_data['images'])
    print(f"Val split ratio: {val_ratio:.1%} ({len(val_split['images'])}/{len(original_data['images'])} images)")
    
    if abs(val_ratio - 0.1) > 0.02:  # Allow 2% tolerance
        print(f"WARNING: Val split ratio {val_ratio:.1%} is not close to target 10%")
    else:
        print("✓ Val split ratio is close to target 10%")
    
    # Check category distribution
    train_cats = Counter(ann['category_id'] for ann in train_split['annotations'])
    val_cats = Counter(ann['category_id'] for ann in val_split['annotations'])
    
    train_unique_cats = set(train_cats.keys())
    val_unique_cats = set(val_cats.keys())
    
    print(f"\nCategory distribution:")
    print(f"Train: {len(train_unique_cats)} unique categories")
    print(f"Val: {len(val_unique_cats)} unique categories")
    
    missing_in_val = train_unique_cats - val_unique_cats
    missing_in_train = val_unique_cats - train_unique_cats
    
    if missing_in_val:
        print(f"Categories in train but not val: {len(missing_in_val)} (e.g., {list(missing_in_val)[:5]})")
    
    if missing_in_train:
        print(f"Categories in val but not train: {len(missing_in_train)} (e.g., {list(missing_in_train)[:5]})")
    
    print("\n✓ Split validation completed successfully")
    return True

def check_environment():
    print("\n=== Environment Check ===\n")
    
    # Check GPU availability
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
                # Check available memory
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  Memory - Allocated: {allocated:.1f} GB, Reserved: {reserved:.1f} GB")
        else:
            print("WARNING: No CUDA GPUs available")
    except ImportError:
        print("ERROR: PyTorch not available")
        return False
    
    # Check required packages
    required_packages = {
        'ultralytics': '8.1.0',
        'torchvision': None,
        'timm': '0.9.12',
        'pycocotools': None,
        'numpy': None,
        'scipy': None,
        'scikit-learn': None
    }
    
    print("\nPackage versions:")
    missing_packages = []
    
    for package, expected_version in required_packages.items():
        try:
            if package == 'ultralytics':
                import ultralytics
                version = ultralytics.__version__
            elif package == 'torchvision':
                import torchvision
                version = torchvision.__version__
            elif package == 'timm':
                import timm
                version = timm.__version__
            elif package == 'pycocotools':
                import pycocotools
                version = getattr(pycocotools, '__version__', 'unknown')
            elif package == 'numpy':
                import numpy
                version = numpy.__version__
            elif package == 'scipy':
                import scipy
                version = scipy.__version__
            elif package == 'scikit-learn':
                import sklearn
                version = sklearn.__version__
            
            status = "✓"
            if expected_version and version != expected_version:
                status = f"⚠ (expected {expected_version})"
            
            print(f"  {package}: {version} {status}")
            
        except ImportError:
            print(f"  {package}: NOT INSTALLED ❌")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nERROR: Missing packages: {missing_packages}")
        return False
    
    # Test ultralytics import specifically
    try:
        from ultralytics import YOLO
        print("\n✓ Ultralytics YOLO import successful")
    except ImportError as e:
        print(f"\nERROR: Cannot import YOLO from ultralytics: {e}")
        return False
    
    # Test pycocotools
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        print("✓ pycocotools import successful")
    except ImportError as e:
        print(f"ERROR: Cannot import from pycocotools: {e}")
        return False
    
    print("\n✓ Environment check completed successfully")
    return True

def main():
    print("=== Step 2: Validate Splits and Check Environment ===\n")
    
    splits_valid = validate_splits()
    env_ready = check_environment()
    
    print(f"\n=== Summary ===\n")
    print(f"Splits valid: {'✓' if splits_valid else '❌'}")
    print(f"Environment ready: {'✓' if env_ready else '❌'}")
    
    if splits_valid and env_ready:
        print("\n🎉 Ready to proceed with experiments!")
    else:
        print("\n⚠ Issues found that need to be resolved before proceeding")

if __name__ == "__main__":
    main()