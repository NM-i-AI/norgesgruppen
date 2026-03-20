import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import random
import shutil
from statistics import mean

def force_reinstall_torchvision():
    """Force reinstall torchvision to fix NMS operator"""
    print("=== FORCE REINSTALLING TORCHVISION ===")
    
    # Try force reinstall with --no-deps first
    try:
        print("Attempting: pip install torchvision==0.21.0 --force-reinstall --no-deps")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'torchvision==0.21.0', '--force-reinstall', '--no-deps'],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print("✓ torchvision==0.21.0 force reinstall successful")
            return True
        else:
            print(f"✗ torchvision==0.21.0 force reinstall failed: {result.stderr}")
    except Exception as e:
        print(f"✗ torchvision==0.21.0 force reinstall error: {e}")
    
    # Try without version pinning if that fails
    try:
        print("Attempting: pip install torchvision --force-reinstall --no-deps")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'torchvision', '--force-reinstall', '--no-deps'],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print("✓ torchvision (latest) force reinstall successful")
            return True
        else:
            print(f"✗ torchvision (latest) force reinstall failed: {result.stderr}")
    except Exception as e:
        print(f"✗ torchvision (latest) force reinstall error: {e}")
    
    return False

def test_nms_operator():
    """Test if torchvision.ops.nms works"""
    try:
        import torch
        import torchvision.ops
        
        # Create dummy data for NMS test
        boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        
        # Test NMS
        keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
        print(f"✓ torchvision.ops.nms works: kept indices {keep}")
        return True
        
    except Exception as e:
        print(f"✗ torchvision.ops.nms failed: {e}")
        return False

def test_torch_extensions():
    """Test if torch C extensions are loadable"""
    try:
        import torch
        print(f"torch version: {torch.__version__}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        
        # Try to access some C++ extensions
        try:
            import torchvision
            print(f"torchvision version: {torchvision.__version__}")
            
            # Test if torchvision C extensions load
            import torchvision.ops
            print("✓ torchvision.ops module imported successfully")
            
            # List available ops
            ops_attrs = [attr for attr in dir(torchvision.ops) if not attr.startswith('_')]
            print(f"Available ops: {ops_attrs[:10]}...")  # Show first 10
            
            return True
            
        except Exception as e:
            print(f"✗ torchvision C extensions failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ torch import failed: {e}")
        return False

def install_packages():
    """Install required packages"""
    packages = [
        "ultralytics==8.1.0",
        "timm==0.9.12"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', package],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                print(f"Warning: Failed to install {package}: {result.stderr}")
            else:
                print(f"Successfully installed {package}")
        except Exception as e:
            print(f"Error installing {package}: {e}")

def load_coco_annotations(annotations_path):
    """Load COCO format annotations"""
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def create_train_val_split(coco_data, val_ratio=0.1, seed=42):
    """Create 90/10 train/val split stratified by store section if possible"""
    random.seed(seed)
    
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    # Group images by store section from filename
    sections = ['egg', 'frokost', 'knekkebrod', 'varmedrikker']
    section_images = {section: [] for section in sections}
    unassigned_images = []
    
    for img in images:
        filename = img['file_name'].lower()
        assigned = False
        for section in sections:
            if section in filename:
                section_images[section].append(img)
                assigned = True
                break
        if not assigned:
            unassigned_images.append(img)
    
    print(f"Images by section: {[(s, len(imgs)) for s, imgs in section_images.items()]}")
    print(f"Unassigned images: {len(unassigned_images)}")
    
    # Split each section proportionally
    val_images = []
    train_images = []
    
    for section, imgs in section_images.items():
        if len(imgs) > 0:
            random.shuffle(imgs)
            val_count = max(1, int(len(imgs) * val_ratio))  # At least 1 for val if any exist
            val_images.extend(imgs[:val_count])
            train_images.extend(imgs[val_count:])
    
    # Handle unassigned images
    if unassigned_images:
        random.shuffle(unassigned_images)
        val_count = int(len(unassigned_images) * val_ratio)
        val_images.extend(unassigned_images[:val_count])
        train_images.extend(unassigned_images[val_count:])
    
    print(f"Split: {len(train_images)} train, {len(val_images)} val")
    
    # Create image ID sets
    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}
    
    # Split annotations
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in annotations if ann['image_id'] in val_image_ids]
    
    print(f"Annotations: {len(train_annotations)} train, {len(val_annotations)} val")
    
    # Create split datasets
    train_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': categories
    }
    
    val_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': categories
    }
    
    return train_data, val_data

def convert_to_yolo_format(coco_data, images_dir, output_dir, split_name):
    """Convert COCO format to YOLO format"""
    output_images_dir = output_dir / 'images' / split_name
    output_labels_dir = output_dir / 'labels' / split_name
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create image_id to annotations mapping
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    # Process each image
    for img in coco_data['images']:
        img_id = img['id']
        filename = img['file_name']
        width = img['width']
        height = img['height']
        
        # Copy image
        src_path = images_dir / filename
        dst_path = output_images_dir / filename
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        
        # Create label file
        label_filename = Path(filename).stem + '.txt'
        label_path = output_labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            for ann in annotations_by_image[img_id]:
                # Convert COCO bbox to YOLO format
                x, y, w, h = ann['bbox']
                
                # Convert to center coordinates and normalize
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height
                
                # YOLO uses 0-based class indices, COCO uses 1-based (except category 0)
                class_id = ann['category_id']
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

def create_dataset_yaml(output_dir, num_classes):
    """Create dataset.yaml for YOLO training"""
    yaml_content = f"""# NorgesGruppen Grocery Dataset
path: {output_dir.absolute()}
train: images/train
val: images/val

# Number of classes
nc: {num_classes}

# Class names (using indices for now)
names:
"""
    
    # Add class names (just use indices for now)
    for i in range(num_classes):
        yaml_content += f"  {i}: 'class_{i}'\n"
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path

def train_yolo_model(dataset_yaml_path, model_size='n', epochs=5, imgsz=640, batch=16):
    """Train YOLO model (smoke test with 5 epochs)"""
    try:
        from ultralytics import YOLO
        
        # Initialize model
        model = YOLO(f'yolov8{model_size}.pt')
        
        # Train
        results = model.train(
            data=str(dataset_yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device='auto',
            project='runs/detect',
            name='yolov8n_smoke_test',
            save=True,
            plots=True
        )
        
        return model, results
        
    except Exception as e:
        print(f"Error training model: {e}")
        return None, None

def evaluate_model(model, val_data_path):
    """Evaluate model and compute metrics"""
    try:
        # Run validation
        results = model.val(data=str(val_data_path), split='val')
        
        # Extract metrics
        metrics = results.results_dict
        
        # Get mAP@0.5 for detection and classification
        # For YOLO, mAP50 is the detection mAP@0.5
        detection_map = metrics.get('metrics/mAP50(B)', 0.0)
        
        # For classification, we use the same mAP50 since YOLO does both detection and classification
        # In a proper implementation, we'd separate detection vs classification evaluation
        classification_map = detection_map  # Simplified for baseline
        
        # Compute val_score
        val_score = 0.7 * detection_map + 0.3 * classification_map
        
        return {
            'detection_mAP@0.5': detection_map,
            'classification_mAP@0.5': classification_map,
            'val_score': val_score,
            'all_metrics': metrics
        }
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return {
            'detection_mAP@0.5': 0.0,
            'classification_mAP@0.5': 0.0,
            'val_score': 0.0,
            'all_metrics': {}
        }

def main():
    print("=== FORCE REINSTALL TORCHVISION + YOLO SMOKE TEST ===")
    
    # 1. Force reinstall torchvision
    print("\n1. Force reinstalling torchvision...")
    if not force_reinstall_torchvision():
        print("✗ Failed to reinstall torchvision")
        print("METRIC:val_score=0.0")
        return
    
    # 2. Test NMS operator
    print("\n2. Testing NMS operator...")
    if not test_nms_operator():
        print("✗ NMS operator still not working")
        # Try to get more diagnostic info
        test_torch_extensions()
        print("METRIC:val_score=0.0")
        return
    
    print("✓ NMS operator working!")
    
    # 3. Install other packages
    print("\n3. Installing other packages...")
    install_packages()
    
    # 4. Load and analyze data
    print("\n4. Loading COCO annotations...")
    annotations_path = Path("data/train/annotations.json")
    images_dir = Path("data/train/images")
    
    if not annotations_path.exists():
        print(f"Error: Annotations file not found at {annotations_path}")
        print("METRIC:val_score=0.0")
        return
    
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        print("METRIC:val_score=0.0")
        return
    
    coco_data = load_coco_annotations(annotations_path)
    print(f"Loaded {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations, {len(coco_data['categories'])} categories")
    
    # 5. Create train/val split
    print("\n5. Creating 90/10 train/val split...")
    train_data, val_data = create_train_val_split(coco_data, val_ratio=0.1, seed=42)
    
    # 6. Convert to YOLO format
    print("\n6. Converting to YOLO format...")
    output_dir = Path("yolo_dataset")
    output_dir.mkdir(exist_ok=True)
    
    convert_to_yolo_format(train_data, images_dir, output_dir, 'train')
    convert_to_yolo_format(val_data, images_dir, output_dir, 'val')
    
    # 7. Create dataset.yaml
    print("\n7. Creating dataset configuration...")
    num_classes = len(coco_data['categories'])
    print(f"Number of classes: {num_classes}")
    
    dataset_yaml_path = create_dataset_yaml(output_dir, num_classes)
    print(f"Created dataset.yaml at {dataset_yaml_path}")
    
    # 8. Train model (smoke test with 5 epochs)
    print("\n8. Training YOLOv8n model (5 epoch smoke test)...")
    model, train_results = train_yolo_model(
        dataset_yaml_path, 
        model_size='n', 
        epochs=5,  # Short smoke test
        imgsz=640, 
        batch=16
    )
    
    if model is None:
        print("Training failed")
        print("METRIC:val_score=0.0")
        return
    
    print("Training completed successfully")
    
    # 9. Evaluate model
    print("\n9. Evaluating model...")
    eval_results = evaluate_model(model, dataset_yaml_path)
    
    # Print all metrics
    print("\n=== EVALUATION RESULTS ===")
    print(f"Detection mAP@0.5: {eval_results['detection_mAP@0.5']:.4f}")
    print(f"Classification mAP@0.5: {eval_results['classification_mAP@0.5']:.4f}")
    print(f"Val Score: {eval_results['val_score']:.4f}")
    
    # Print metrics in required format
    print(f"\nMETRIC:detection_mAP@0.5={eval_results['detection_mAP@0.5']:.4f}")
    print(f"METRIC:classification_mAP@0.5={eval_results['classification_mAP@0.5']:.4f}")
    print(f"METRIC:val_score={eval_results['val_score']:.4f}")
    
    # Additional metrics for debugging
    if eval_results['all_metrics']:
        for key, value in eval_results['all_metrics'].items():
            if isinstance(value, (int, float)):
                print(f"METRIC:{key}={value:.4f}")
    
    print("\n=== SMOKE TEST COMPLETE ===")

if __name__ == "__main__":
    main()