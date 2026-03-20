import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

def load_annotations():
    """Load and parse the COCO annotations file"""
    with open('data/train/annotations.json', 'r') as f:
        data = json.load(f)
    return data

def load_metadata():
    """Load product metadata"""
    with open('data/metadata.json', 'r') as f:
        data = json.load(f)
    return data

def analyze_images(coco_data):
    """Analyze image statistics"""
    images = coco_data['images']
    print(f"\n=== IMAGE ANALYSIS ===")
    print(f"Total images: {len(images)}")
    
    # Image dimensions
    widths = [img['width'] for img in images]
    heights = [img['height'] for img in images]
    
    print(f"\nImage dimensions:")
    print(f"  Width: min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}")
    print(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}")
    
    # Check for store sections in filenames
    filenames = [img['file_name'] for img in images]
    print(f"\nSample filenames:")
    for i, fname in enumerate(filenames[:10]):
        print(f"  {fname}")
    
    # Try to extract store sections from filenames
    sections = set()
    for fname in filenames:
        # Look for common patterns that might indicate store sections
        fname_lower = fname.lower()
        if 'egg' in fname_lower:
            sections.add('Egg')
        elif 'frokost' in fname_lower:
            sections.add('Frokost')
        elif 'knekkebrod' in fname_lower or 'knekke' in fname_lower:
            sections.add('Knekkebrod')
        elif 'varme' in fname_lower or 'drikk' in fname_lower:
            sections.add('Varmedrikker')
    
    print(f"\nDetected store sections from filenames: {sections}")
    
    return images

def analyze_annotations(coco_data):
    """Analyze annotation statistics"""
    annotations = coco_data['annotations']
    images = coco_data['images']
    categories = coco_data['categories']
    
    print(f"\n=== ANNOTATION ANALYSIS ===")
    print(f"Total annotations: {len(annotations)}")
    print(f"Total categories: {len(categories)}")
    
    # Annotations per image
    image_id_to_anns = defaultdict(list)
    for ann in annotations:
        image_id_to_anns[ann['image_id']].append(ann)
    
    anns_per_image = [len(anns) for anns in image_id_to_anns.values()]
    print(f"\nAnnotations per image:")
    print(f"  Min: {min(anns_per_image)}")
    print(f"  Max: {max(anns_per_image)}")
    print(f"  Mean: {np.mean(anns_per_image):.1f}")
    print(f"  Median: {np.median(anns_per_image):.1f}")
    
    # Category frequency distribution
    category_counts = Counter(ann['category_id'] for ann in annotations)
    print(f"\nCategory distribution:")
    print(f"  Categories with 1 annotation: {sum(1 for count in category_counts.values() if count == 1)}")
    print(f"  Categories with 2-5 annotations: {sum(1 for count in category_counts.values() if 2 <= count <= 5)}")
    print(f"  Categories with 6-20 annotations: {sum(1 for count in category_counts.values() if 6 <= count <= 20)}")
    print(f"  Categories with >20 annotations: {sum(1 for count in category_counts.values() if count > 20)}")
    
    # Most and least frequent categories
    most_common = category_counts.most_common(10)
    least_common = [(cat_id, count) for cat_id, count in category_counts.items() if count <= 5]
    
    print(f"\nMost frequent categories:")
    for cat_id, count in most_common:
        cat_name = next((cat['name'] for cat in categories if cat['id'] == cat_id), 'Unknown')
        print(f"  Category {cat_id} ({cat_name}): {count} annotations")
    
    print(f"\nRare categories (≤5 annotations): {len(least_common)}")
    if len(least_common) <= 20:  # Show all if not too many
        for cat_id, count in sorted(least_common, key=lambda x: x[1]):
            cat_name = next((cat['name'] for cat in categories if cat['id'] == cat_id), 'Unknown')
            print(f"  Category {cat_id} ({cat_name}): {count} annotations")
    
    # Check category 356 (unknown_product)
    unknown_count = category_counts.get(356, 0)
    print(f"\nCategory 356 (unknown_product): {unknown_count} annotations")
    
    # Bbox analysis
    bboxes = [ann['bbox'] for ann in annotations]
    areas = [bbox[2] * bbox[3] for bbox in bboxes]  # width * height
    
    print(f"\nBounding box analysis:")
    print(f"  Area: min={min(areas):.1f}, max={max(areas):.1f}, mean={np.mean(areas):.1f}")
    print(f"  Width: min={min(bbox[2] for bbox in bboxes):.1f}, max={max(bbox[2] for bbox in bboxes):.1f}")
    print(f"  Height: min={min(bbox[3] for bbox in bboxes):.1f}, max={max(bbox[3] for bbox in bboxes):.1f}")
    
    # Check for corrected flag and iscrowd
    corrected_count = sum(1 for ann in annotations if ann.get('corrected', False))
    iscrowd_count = sum(1 for ann in annotations if ann.get('iscrowd', 0) == 1)
    
    print(f"\nAnnotation flags:")
    print(f"  Corrected annotations: {corrected_count} ({corrected_count/len(annotations)*100:.1f}%)")
    print(f"  Crowd annotations: {iscrowd_count} ({iscrowd_count/len(annotations)*100:.1f}%)")
    
    return annotations, category_counts

def analyze_categories(coco_data):
    """Analyze category information"""
    categories = coco_data['categories']
    
    print(f"\n=== CATEGORY ANALYSIS ===")
    print(f"Total categories: {len(categories)}")
    
    # Check category ID range
    cat_ids = [cat['id'] for cat in categories]
    print(f"Category ID range: {min(cat_ids)} to {max(cat_ids)}")
    
    # Check for category 356
    cat_356 = next((cat for cat in categories if cat['id'] == 356), None)
    if cat_356:
        print(f"Category 356: {cat_356}")
    
    # Sample categories
    print(f"\nSample categories:")
    for i, cat in enumerate(categories[:10]):
        print(f"  {cat['id']}: {cat['name']}")
    
    return categories

def analyze_products_directory():
    """Analyze the products reference images directory"""
    products_dir = Path('data/products')
    
    print(f"\n=== PRODUCTS DIRECTORY ANALYSIS ===")
    
    if not products_dir.exists():
        print("Products directory not found!")
        return
    
    # Count product directories
    product_dirs = [d for d in products_dir.iterdir() if d.is_dir()]
    print(f"Total product directories: {len(product_dirs)}")
    
    # Analyze images per product
    images_per_product = []
    image_types = Counter()
    
    for i, product_dir in enumerate(product_dirs[:20]):  # Sample first 20
        images = list(product_dir.glob('*.jpg')) + list(product_dir.glob('*.jpeg')) + list(product_dir.glob('*.png'))
        images_per_product.append(len(images))
        
        for img in images:
            image_types[img.stem] += 1  # Count by filename (main, front, etc.)
        
        if i < 5:  # Show details for first 5
            print(f"  Product {product_dir.name}: {len(images)} images")
            for img in images:
                print(f"    {img.name}")
    
    if images_per_product:
        print(f"\nImages per product (sample of {len(images_per_product)}):")
        print(f"  Min: {min(images_per_product)}")
        print(f"  Max: {max(images_per_product)}")
        print(f"  Mean: {np.mean(images_per_product):.1f}")
    
    print(f"\nCommon image types:")
    for img_type, count in image_types.most_common(10):
        print(f"  {img_type}: {count} images")

def analyze_metadata(metadata):
    """Analyze the metadata.json structure"""
    print(f"\n=== METADATA ANALYSIS ===")
    
    if not metadata:
        print("No metadata loaded")
        return
    
    print(f"Metadata keys: {list(metadata.keys())}")
    
    # If it's a list of products
    if isinstance(metadata, list):
        print(f"Total products in metadata: {len(metadata)}")
        
        if len(metadata) > 0:
            sample_product = metadata[0]
            print(f"\nSample product structure:")
            for key, value in sample_product.items():
                if isinstance(value, (str, int, float, bool)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
    
    # If it's a dict with product info
    elif isinstance(metadata, dict):
        for key, value in metadata.items():
            if isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
                if len(value) > 0 and isinstance(value[0], dict):
                    print(f"    Sample item keys: {list(value[0].keys())}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} keys")
            else:
                print(f"  {key}: {type(value)} = {value}")

def main():
    print("=== DATASET DEEP DIVE ANALYSIS ===")
    
    try:
        # Load data
        print("Loading annotations...")
        coco_data = load_annotations()
        
        print("Loading metadata...")
        metadata = load_metadata()
        
        # Run analyses
        images = analyze_images(coco_data)
        annotations, category_counts = analyze_annotations(coco_data)
        categories = analyze_categories(coco_data)
        analyze_products_directory()
        analyze_metadata(metadata)
        
        # Summary statistics
        print(f"\n=== SUMMARY ===")
        print(f"Dataset size: {len(images)} images, {len(annotations)} annotations, {len(categories)} categories")
        print(f"Average annotations per image: {len(annotations)/len(images):.1f}")
        print(f"Categories with <10 annotations: {sum(1 for count in category_counts.values() if count < 10)}")
        print(f"This is a dense, few-shot detection problem with high class imbalance")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()