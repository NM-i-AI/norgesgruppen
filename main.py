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
from utils import create_train_val_split

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
    """Build embedding gallery from reference product images.
    
    Returns:
        tuple: (embeddings_array, category_ids, image_paths)
    """
    products_dir = Path("data/products")
    
    if not products_dir.exists():
        print(f"Products directory not found: {products_dir}")
        return None, None, None
    
    # Load metadata to get category mappings
    metadata_path = Path("data/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    embeddings = []
    category_ids = []
    image_paths = []
    
    print("Building reference gallery from product images...")
    
    # Process each product directory
    for product_dir in sorted(products_dir.iterdir()):
        if not product_dir.is_dir():
            continue
            
        product_code = product_dir.name
        
        # Get category_id from metadata if available
        category_id = None
        if product_code in metadata:
            category_id = metadata[product_code].get('category_id')
        
        if category_id is None:
            # Skip products without category mapping
            continue
        
        # Look for reference images (prioritize main.jpg, front.jpg)
        reference_images = []
        for img_name in ['main.jpg', 'front.jpg', 'back.jpg', 'side.jpg']:
            img_path = product_dir / img_name
            if img_path.exists():
                reference_images.append(img_path)
        
        # Process each reference image
        for img_path in reference_images:
            try:
                # Extract features
                features = extract_dinov2_features(dinov2_model, img_path)
                
                embeddings.append(features)
                category_ids.append(category_id)
                image_paths.append(str(img_path))
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    if not embeddings:
        print("No reference embeddings extracted")
        return None, None, None
    
    embeddings_array = np.vstack(embeddings)
    print(f"Built gallery: {len(embeddings)} embeddings from {len(set(category_ids))} categories")
    
    return embeddings_array, category_ids, image_paths

def evaluate_knn_classification(dinov2_model, gallery_embeddings, gallery_categories, val_image_ids, k=5):
    """Evaluate kNN classification on validation GT crops.
    
    Args:
        dinov2_model: DINOv2 model
        gallery_embeddings: Reference embeddings array
        gallery_categories: Category IDs for gallery
        val_image_ids: Validation image IDs
        k: Number of nearest neighbors
    
    Returns:
        dict: Classification metrics
    """
    # Load annotations
    annotations_path = Path("data/train/annotations.json")
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Get validation annotations (GT crops)
    val_annotations = [ann for ann in coco_data['annotations'] 
                      if ann['image_id'] in val_image_ids]
    
    # Create image_id to filename mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Build kNN index
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(gallery_embeddings)
    
    print(f"Evaluating kNN classification on {len(val_annotations)} GT crops...")
    
    correct_predictions = 0
    total_predictions = 0
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    images_dir = Path("data/train/images")
    
    for ann in val_annotations:
        try:
            # Get image path
            img_info = image_info[ann['image_id']]
            img_path = images_dir / img_info['file_name']
            
            if not img_path.exists():
                continue
            
            # Extract crop features
            bbox = ann['bbox']  # [x, y, w, h]
            crop_features = extract_dinov2_features(dinov2_model, img_path, bbox)
            
            # Find k nearest neighbors
            distances, indices = knn.kneighbors([crop_features])
            
            # Get neighbor categories
            neighbor_categories = [gallery_categories[idx] for idx in indices[0]]
            
            # Predict category (majority vote)
            category_counts = Counter(neighbor_categories)
            predicted_category = category_counts.most_common(1)[0][0]
            
            # Check if correct
            true_category = ann['category_id']
            is_correct = (predicted_category == true_category)
            
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            # Update per-category stats
            category_stats[true_category]['total'] += 1
            if is_correct:
                category_stats[true_category]['correct'] += 1
                
        except Exception as e:
            print(f"Error processing annotation {ann['id']}: {e}")
            continue
    
    # Calculate metrics
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # Per-category accuracy
    category_accuracies = {}
    for cat_id, stats in category_stats.items():
        if stats['total'] > 0:
            category_accuracies[cat_id] = stats['correct'] / stats['total']
    
    # Categories with samples
    categories_with_samples = len([acc for acc in category_accuracies.values() if acc >= 0])
    
    results = {
        'overall_accuracy': overall_accuracy,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions,
        'categories_evaluated': categories_with_samples,
        'category_accuracies': category_accuracies
    }
    
    return results

def main():
    print("=== DINOv2 EMBEDDING GALLERY + kNN CLASSIFICATION ===\n")
    
    try:
        # Step 1: Load DINOv2 model
        print("1. Loading DINOv2-base model...")
        dinov2_model = load_dinov2_model()
        if dinov2_model is None:
            raise Exception("Failed to load DINOv2 model")
        print("✓ DINOv2-base model loaded")
        
        # Step 2: Create train/val split
        print("\n2. Creating train/val split...")
        annotations_path = Path("data/train/annotations.json")
        train_image_ids, val_image_ids = create_train_val_split(annotations_path, val_ratio=0.1, seed=42)
        print(f"✓ Split: {len(train_image_ids)} train, {len(val_image_ids)} val images")
        
        # Step 3: Build reference gallery
        print("\n3. Building reference embedding gallery...")
        gallery_embeddings, gallery_categories, gallery_paths = build_reference_gallery(dinov2_model)
        
        if gallery_embeddings is None:
            raise Exception("Failed to build reference gallery")
        
        print(f"✓ Gallery built: {gallery_embeddings.shape[0]} embeddings")
        print(f"  - Embedding dimension: {gallery_embeddings.shape[1]}")
        print(f"  - Unique categories: {len(set(gallery_categories))}")
        
        # Step 4: Evaluate kNN classification on GT crops
        print("\n4. Evaluating kNN classification on validation GT crops...")
        
        # Test different k values
        k_values = [1, 3, 5, 7]
        best_k = 1
        best_accuracy = 0.0
        
        for k in k_values:
            print(f"\n  Testing k={k}...")
            results = evaluate_knn_classification(
                dinov2_model, gallery_embeddings, gallery_categories, val_image_ids, k=k
            )
            
            accuracy = results['overall_accuracy']
            print(f"    Accuracy: {accuracy:.4f} ({results['correct_predictions']}/{results['total_predictions']})")
            print(f"    Categories evaluated: {results['categories_evaluated']}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
        
        print(f"\n=== kNN CLASSIFICATION RESULTS ===\n")
        print(f"Best k: {best_k}")
        print(f"Best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.1f}%)")
        
        # Detailed results for best k
        print(f"\nDetailed results for k={best_k}:")
        best_results = evaluate_knn_classification(
            dinov2_model, gallery_embeddings, gallery_categories, val_image_ids, k=best_k
        )
        
        print(f"  Total predictions: {best_results['total_predictions']}")
        print(f"  Correct predictions: {best_results['correct_predictions']}")
        print(f"  Overall accuracy: {best_results['overall_accuracy']:.4f}")
        print(f"  Categories with samples: {best_results['categories_evaluated']}")
        
        # Show per-category accuracy for categories with >5 samples
        category_accs = best_results['category_accuracies']
        frequent_categories = {cat_id: acc for cat_id, acc in category_accs.items() 
                             if best_results['total_predictions'] > 0}
        
        if frequent_categories:
            print(f"\n  Sample per-category accuracies:")
            sorted_cats = sorted(frequent_categories.items(), key=lambda x: x[1], reverse=True)[:10]
            for cat_id, acc in sorted_cats:
                print(f"    Category {cat_id}: {acc:.3f}")
        
        # Success criteria check
        success_threshold = 0.30  # 30% top-1 accuracy
        success = best_accuracy >= success_threshold
        
        print(f"\nSuccess criteria: kNN accuracy >= {success_threshold:.1%}")
        if success:
            print(f"✓ SUCCESS: {best_accuracy:.1%} >= {success_threshold:.1%}")
        else:
            print(f"⚠ BELOW THRESHOLD: {best_accuracy:.1%} < {success_threshold:.1%}")
        
        # Output metrics
        print(f"\nMETRIC:knn_accuracy={best_accuracy:.4f}")
        print(f"METRIC:best_k={best_k}")
        print(f"METRIC:total_predictions={best_results['total_predictions']}")
        print(f"METRIC:correct_predictions={best_results['correct_predictions']}")
        print(f"METRIC:categories_evaluated={best_results['categories_evaluated']}")
        print(f"METRIC:gallery_size={gallery_embeddings.shape[0]}")
        print(f"METRIC:gallery_categories={len(set(gallery_categories))}")
        print(f"METRIC:success_criteria_met={1.0 if success else 0.0}")
        print(f"METRIC:embedding_dimension={gallery_embeddings.shape[1]}")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"METRIC:knn_accuracy=0.0")
        print(f"METRIC:experiment_failed=1.0")

if __name__ == "__main__":
    main()