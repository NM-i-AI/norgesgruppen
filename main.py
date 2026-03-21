# Fix numpy.trapz compatibility before importing ultralytics
import numpy as np
if not hasattr(np, 'trapz'):
    # numpy >= 2.0 removed trapz, use trapezoid instead
    if hasattr(np, 'trapezoid'):
        np.trapz = np.trapezoid
    else:
        # Fallback to scipy if numpy.trapezoid also missing
        from scipy.integrate import trapezoid
        np.trapz = trapezoid

import torch

# Monkey-patch torch.load to fix ultralytics compatibility with PyTorch 2.6
# PyTorch 2.6 changed default weights_only=True for security, but ultralytics expects False
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Now import other packages
import json
import os
import shutil
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, Counter
from utils import create_train_val_split, save_coco_split, evaluate_model
from ultralytics import YOLO

def load_dinov2_model():
    """Load DINOv2-base model for feature extraction."""
    try:
        import timm
        # Load DINOv2-base model
        model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading DINOv2 model: {e}")
        return None

def extract_dinov2_features(model, image_path, bbox=None):
    """Extract DINOv2 features from an image or image crop.
    
    Args:
        model: DINOv2 model
        image_path: Path to image
        bbox: Optional [x, y, w, h] bbox to crop (COCO format)
    
    Returns:
        numpy array: 768-dim feature vector
    """
    # DINOv2 preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and optionally crop image
    image = Image.open(image_path).convert('RGB')
    
    if bbox is not None:
        x, y, w, h = bbox
        # Ensure crop is within image bounds
        x = max(0, int(x))
        y = max(0, int(y))
        w = min(image.width - x, int(w))
        h = min(image.height - y, int(h))
        image = image.crop((x, y, x + w, y + h))
    
    # Preprocess and extract features
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        features = model.forward_features(input_tensor)
        # Get CLS token (first token)
        cls_features = features[:, 0, :].cpu().numpy()
    
    return cls_features.flatten()

def build_reference_gallery(dinov2_model):
    """Build embedding gallery from reference product images and training crops.
    
    Returns:
        tuple: (embeddings_array, category_ids, image_paths)
    """
    # Load metadata to get category mappings
    metadata_path = Path("data/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        print("Warning: metadata.json not found")
        metadata = {}
    
    embeddings = []
    category_ids = []
    image_paths = []
    
    print("Building reference gallery...")
    
    # 1. Add reference product images if available
    products_dir = Path("data/products")
    if products_dir.exists():
        print("  Adding reference product images...")
        for product_dir in sorted(products_dir.iterdir()):
            if not product_dir.is_dir():
                continue
                
            product_code = product_dir.name
            
            # Get category_id from metadata if available
            category_id = None
            if product_code in metadata:
                category_id = metadata[product_code].get('category_id')
            
            if category_id is None:
                continue
            
            # Look for reference images (prioritize main.jpg, front.jpg)
            reference_images = []
            for img_name in ['main.jpg', 'front.jpg', 'back.jpg', 'side.jpg']:
                img_path = product_dir / img_name
                if img_path.exists():
                    reference_images.append(img_path)
            
            # Process each reference image (repeat main/front 3x for higher weight)
            for img_path in reference_images:
                try:
                    features = extract_dinov2_features(dinov2_model, img_path)
                    
                    # Add multiple copies for main/front views
                    repeat_count = 3 if img_path.name in ['main.jpg', 'front.jpg'] else 1
                    for _ in range(repeat_count):
                        embeddings.append(features)
                        category_ids.append(category_id)
                        image_paths.append(str(img_path))
                        
                except Exception as e:
                    print(f"    Error processing {img_path}: {e}")
                    continue
    
    # 2. Add training crop embeddings (weighted 2.0x)
    print("  Adding training crop embeddings...")
    annotations_path = Path("data/train/annotations.json")
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to filename mapping
    image_info = {img['id']: img for img in coco_data['images']}
    images_dir = Path("data/train/images")
    
    # Sample training annotations (limit to avoid memory issues)
    train_annotations = coco_data['annotations']
    if len(train_annotations) > 5000:  # Limit for memory
        import random
        random.seed(42)
        train_annotations = random.sample(train_annotations, 5000)
    
    for ann in train_annotations:
        try:
            img_info = image_info[ann['image_id']]
            img_path = images_dir / img_info['file_name']
            
            if not img_path.exists():
                continue
            
            # Extract crop features
            bbox = ann['bbox']  # [x, y, w, h]
            features = extract_dinov2_features(dinov2_model, img_path, bbox)
            
            # Add with 2x weight for training crops
            for _ in range(2):
                embeddings.append(features)
                category_ids.append(ann['category_id'])
                image_paths.append(f"train_crop_{ann['id']}")
                
        except Exception as e:
            continue  # Skip problematic annotations
    
    if not embeddings:
        print("No embeddings extracted")
        return None, None, None
    
    embeddings_array = np.vstack(embeddings)
    print(f"Gallery built: {len(embeddings)} embeddings from {len(set(category_ids))} categories")
    
    return embeddings_array, category_ids, image_paths

def train_single_class_detector():
    """Train single-class YOLO detector (nc=1)."""
    print("Training single-class detector...")
    
    # Prepare dataset with nc=1 (all categories mapped to class 0)
    annotations_path = Path("data/train/annotations.json")
    train_image_ids, val_image_ids = create_train_val_split(annotations_path, val_ratio=0.1, seed=42)
    
    # Load and modify annotations for single-class
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Map all categories to class 0
    for ann in coco_data['annotations']:
        ann['category_id'] = 0
    
    # Update categories to single class
    coco_data['categories'] = [{'id': 0, 'name': 'product'}]
    
    # Save train/val splits
    save_coco_split(coco_data, train_image_ids, "train_split.json")
    save_coco_split(coco_data, val_image_ids, "val_split.json")
    
    # Convert to YOLO format
    from ultralytics.data.converter import convert_coco
    
    # Clean up any existing dataset
    if Path("grocery_yolo").exists():
        shutil.rmtree("grocery_yolo")
    
    # Convert COCO to YOLO format
    convert_coco(
        labels_dir=".",
        save_dir="grocery_yolo",
        use_segments=False,
        use_keypoints=False,
        cls91to80=False
    )
    
    # Create dataset YAML
    yaml_content = f"""path: {Path('grocery_yolo').absolute()}
train: train/images
val: val/images
nc: 1
names:
  0: product
"""
    
    with open("grocery_single_class.yaml", "w") as f:
        f.write(yaml_content)
    
    # Train model
    model = YOLO('yolov8m.pt')  # Use YOLOv8m as good balance
    
    results = model.train(
        data="grocery_single_class.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device=[0, 1],  # Use both GPUs
        patience=50,
        save=True,
        cache=True,
        close_mosaic=10
    )
    
    return model

def run_two_stage_pipeline(detector_model, dinov2_model, gallery_embeddings, gallery_categories, val_image_ids, conf_threshold=0.25, nms_iou=0.7):
    """Run two-stage detection + classification pipeline.
    
    Args:
        detector_model: Trained single-class YOLO model
        dinov2_model: DINOv2 model for feature extraction
        gallery_embeddings: Reference embeddings array
        gallery_categories: Category IDs for gallery
        val_image_ids: Validation image IDs
        conf_threshold: Detection confidence threshold
        nms_iou: NMS IoU threshold
    
    Returns:
        list: Predictions in COCO format
    """
    # Load annotations for image info
    annotations_path = Path("data/train/annotations.json")
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    image_info = {img['id']: img for img in coco_data['images']}
    images_dir = Path("data/train/images")
    
    # Build kNN index
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(gallery_embeddings)
    
    predictions = []
    
    print(f"Running two-stage pipeline on {len(val_image_ids)} validation images...")
    
    for image_id in val_image_ids:
        try:
            img_info = image_info[image_id]
            img_path = images_dir / img_info['file_name']
            
            if not img_path.exists():
                continue
            
            # Stage 1: Detection
            results = detector_model.predict(
                source=str(img_path),
                conf=conf_threshold,
                iou=nms_iou,
                verbose=False
            )
            
            if not results or len(results) == 0:
                continue
            
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            
            # Stage 2: Classification for each detection
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores = result.boxes.conf.cpu().numpy()
            
            for i, (box, score) in enumerate(zip(boxes, scores)):
                # Convert to COCO format [x, y, w, h]
                x1, y1, x2, y2 = box
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                bbox_coco = [x, y, w, h]
                
                try:
                    # Extract crop features
                    crop_features = extract_dinov2_features(dinov2_model, img_path, bbox_coco)
                    
                    # Find nearest neighbors
                    distances, indices = knn.kneighbors([crop_features])
                    
                    # Get neighbor categories and distances
                    neighbor_categories = [gallery_categories[idx] for idx in indices[0]]
                    neighbor_distances = distances[0]
                    
                    # Weighted voting based on distance (closer = higher weight)
                    category_weights = defaultdict(float)
                    for cat_id, dist in zip(neighbor_categories, neighbor_distances):
                        # Use inverse distance as weight (add small epsilon to avoid division by zero)
                        weight = 1.0 / (dist + 1e-6)
                        category_weights[cat_id] += weight
                    
                    # Predict category with highest weight
                    if category_weights:
                        predicted_category = max(category_weights.items(), key=lambda x: x[1])[0]
                    else:
                        predicted_category = 0  # Fallback to unknown
                    
                    # Create prediction
                    pred = {
                        'image_id': image_id,
                        'category_id': int(predicted_category),
                        'bbox': [float(x) for x in bbox_coco],
                        'score': float(score)
                    }
                    predictions.append(pred)
                    
                except Exception as e:
                    # If classification fails, use category_id=0 (detection only)
                    pred = {
                        'image_id': image_id,
                        'category_id': 0,
                        'bbox': [float(x) for x in bbox_coco],
                        'score': float(score)
                    }
                    predictions.append(pred)
                    
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            continue
    
    return predictions

def main():
    print("=== TWO-STAGE PIPELINE: DETECTOR + DINOv2 CLASSIFIER ===\n")
    
    try:
        # Step 1: Load DINOv2 model
        print("1. Loading DINOv2-base model...")
        dinov2_model = load_dinov2_model()
        if dinov2_model is None:
            raise Exception("Failed to load DINOv2 model")
        print("✓ DINOv2-base model loaded")
        
        # Step 2: Build reference gallery
        print("\n2. Building reference embedding gallery...")
        gallery_embeddings, gallery_categories, gallery_paths = build_reference_gallery(dinov2_model)
        
        if gallery_embeddings is None:
            raise Exception("Failed to build reference gallery")
        
        print(f"✓ Gallery built: {gallery_embeddings.shape[0]} embeddings")
        print(f"  - Unique categories: {len(set(gallery_categories))}")
        
        # Step 3: Train single-class detector
        print("\n3. Training single-class detector...")
        detector_model = train_single_class_detector()
        print("✓ Single-class detector trained")
        
        # Step 4: Create validation split
        print("\n4. Creating validation split...")
        annotations_path = Path("data/train/annotations.json")
        train_image_ids, val_image_ids = create_train_val_split(annotations_path, val_ratio=0.1, seed=42)
        print(f"✓ Split: {len(train_image_ids)} train, {len(val_image_ids)} val images")
        
        # Step 5: Run two-stage pipeline
        print("\n5. Running two-stage pipeline on validation set...")
        
        # Test different confidence thresholds
        conf_thresholds = [0.1, 0.25, 0.4]
        best_conf = 0.25
        best_val_score = 0.0
        best_predictions = []
        
        for conf_threshold in conf_thresholds:
            print(f"\n  Testing conf_threshold={conf_threshold}...")
            
            predictions = run_two_stage_pipeline(
                detector_model, dinov2_model, gallery_embeddings, gallery_categories,
                val_image_ids, conf_threshold=conf_threshold, nms_iou=0.7
            )
            
            print(f"    Generated {len(predictions)} predictions")
            
            # Evaluate
            val_score, detection_map, classification_map = evaluate_model(
                predictions, val_image_ids, annotations_path
            )
            
            print(f"    val_score: {val_score:.4f} (det: {detection_map:.4f}, cls: {classification_map:.4f})")
            
            if val_score > best_val_score:
                best_val_score = val_score
                best_conf = conf_threshold
                best_predictions = predictions
        
        # Final results
        print(f"\n=== TWO-STAGE PIPELINE RESULTS ===\n")
        print(f"Best confidence threshold: {best_conf}")
        print(f"Best val_score: {best_val_score:.4f}")
        
        # Detailed evaluation with best parameters
        val_score, detection_map, classification_map = evaluate_model(
            best_predictions, val_image_ids, annotations_path
        )
        
        print(f"\nDetailed results:")
        print(f"  Detection mAP@0.5: {detection_map:.4f}")
        print(f"  Classification mAP@0.5: {classification_map:.4f}")
        print(f"  Combined val_score: {val_score:.4f}")
        print(f"  Total predictions: {len(best_predictions)}")
        
        # Success criteria
        baseline_score = 0.5843  # Best from exp-011
        success = val_score > baseline_score
        
        print(f"\nSuccess criteria: val_score > {baseline_score:.4f} (best multiclass YOLO)")
        if success:
            print(f"✓ SUCCESS: {val_score:.4f} > {baseline_score:.4f} (+{val_score - baseline_score:.4f})")
        else:
            print(f"⚠ BELOW BASELINE: {val_score:.4f} <= {baseline_score:.4f} ({val_score - baseline_score:.4f})")
        
        # Output metrics
        print(f"\nMETRIC:val_score={val_score:.4f}")
        print(f"METRIC:detection_map={detection_map:.4f}")
        print(f"METRIC:classification_map={classification_map:.4f}")
        print(f"METRIC:best_conf_threshold={best_conf}")
        print(f"METRIC:total_predictions={len(best_predictions)}")
        print(f"METRIC:gallery_size={gallery_embeddings.shape[0]}")
        print(f"METRIC:gallery_categories={len(set(gallery_categories))}")
        print(f"METRIC:success_vs_baseline={1.0 if success else 0.0}")
        print(f"METRIC:improvement_over_baseline={val_score - baseline_score:.4f}")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"METRIC:val_score=0.0")
        print(f"METRIC:experiment_failed=1.0")

if __name__ == "__main__":
    main()