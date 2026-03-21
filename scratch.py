import json
import os
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def main():
    print("=== Deep Annotation Analysis ===")
    
    data_path = Path("data")
    annotations_path = data_path / "train" / "annotations.json"
    
    if not annotations_path.exists():
        print("Annotations file not found!")
        return
    
    with open(annotations_path) as f:
        coco_data = json.load(f)
    
    print(f"\n=== COCO Data Structure ===")
    print(f"Images: {len(coco_data['images'])}")
    print(f"Annotations: {len(coco_data['annotations'])}")
    print(f"Categories: {len(coco_data['categories'])}")
    
    # Examine annotation fields in detail
    print("\n=== Annotation Fields Analysis ===")
    sample_ann = coco_data['annotations'][0]
    print(f"Sample annotation fields: {list(sample_ann.keys())}")
    print(f"Sample annotation: {sample_ann}")
    
    # Check for 'corrected' flag
    corrected_count = 0
    has_corrected_field = False
    for ann in coco_data['annotations']:
        if 'corrected' in ann:
            has_corrected_field = True
            if ann['corrected']:
                corrected_count += 1
    
    if has_corrected_field:
        print(f"\nCorrected annotations: {corrected_count}/{len(coco_data['annotations'])} ({corrected_count/len(coco_data['annotations'])*100:.1f}%)")
    else:
        print("\nNo 'corrected' field found in annotations")
    
    # Bbox format and area analysis
    print("\n=== Bbox Analysis ===")
    areas = []
    widths = []
    heights = []
    aspect_ratios = []
    
    for ann in coco_data['annotations']:
        bbox = ann['bbox']  # [x, y, width, height] in COCO format
        w, h = bbox[2], bbox[3]
        area = w * h
        
        areas.append(area)
        widths.append(w)
        heights.append(h)
        if h > 0:
            aspect_ratios.append(w / h)
    
    print(f"Bbox areas (pixels²):")
    print(f"  Mean: {np.mean(areas):.0f}")
    print(f"  Median: {np.median(areas):.0f}")
    print(f"  Min: {min(areas):.0f}, Max: {max(areas):.0f}")
    print(f"  25th percentile: {np.percentile(areas, 25):.0f}")
    print(f"  75th percentile: {np.percentile(areas, 75):.0f}")
    
    print(f"\nBbox dimensions:")
    print(f"  Width - Mean: {np.mean(widths):.1f}, Median: {np.median(widths):.1f}")
    print(f"  Height - Mean: {np.mean(heights):.1f}, Median: {np.median(heights):.1f}")
    print(f"  Aspect ratio - Mean: {np.mean(aspect_ratios):.2f}, Median: {np.median(aspect_ratios):.2f}")
    
    # Small object analysis (important for detection)
    small_objects = sum(1 for area in areas if area < 32*32)  # COCO small object threshold
    medium_objects = sum(1 for area in areas if 32*32 <= area < 96*96)
    large_objects = sum(1 for area in areas if area >= 96*96)
    
    print(f"\nObject size distribution (COCO thresholds):")
    print(f"  Small (<32²): {small_objects} ({small_objects/len(areas)*100:.1f}%)")
    print(f"  Medium (32²-96²): {medium_objects} ({medium_objects/len(areas)*100:.1f}%)")
    print(f"  Large (≥96²): {large_objects} ({large_objects/len(areas)*100:.1f}%)")
    
    # Image size context
    print("\n=== Image Size Context ===")
    image_sizes = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
    
    # Relative bbox sizes
    relative_areas = []
    for ann in coco_data['annotations']:
        img_w, img_h = image_sizes[ann['image_id']]
        bbox_area = ann['bbox'][2] * ann['bbox'][3]
        img_area = img_w * img_h
        relative_areas.append(bbox_area / img_area)
    
    print(f"Relative bbox areas (% of image):")
    print(f"  Mean: {np.mean(relative_areas)*100:.3f}%")
    print(f"  Median: {np.median(relative_areas)*100:.3f}%")
    print(f"  Min: {min(relative_areas)*100:.4f}%, Max: {max(relative_areas)*100:.2f}%")
    
    # Category analysis
    print("\n=== Category Mapping Analysis ===")
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"Category ID range: {min(categories.keys())} to {max(categories.keys())}")
    
    # Check for category 356 (unknown_product)
    if 356 in categories:
        print(f"Category 356 (unknown_product): {categories[356]}")
        unknown_count = sum(1 for ann in coco_data['annotations'] if ann['category_id'] == 356)
        print(f"  Annotations with category 356: {unknown_count}")
    
    # Category frequency distribution
    cat_counts = Counter(ann['category_id'] for ann in coco_data['annotations'])
    print(f"\nCategory frequency distribution:")
    freq_bins = [1, 2, 3, 5, 10, 20, 50, 100, 200]
    for i in range(len(freq_bins)):
        if i == 0:
            count = sum(1 for c in cat_counts.values() if c == freq_bins[i])
            print(f"  Exactly {freq_bins[i]} annotation: {count} categories")
        else:
            lower = freq_bins[i-1] + 1 if i > 1 else freq_bins[i-1]
            upper = freq_bins[i]
            count = sum(1 for c in cat_counts.values() if lower <= c <= upper)
            print(f"  {lower}-{upper} annotations: {count} categories")
    
    # Categories with most annotations
    print(f"\nTop 15 categories by annotation count:")
    for cat_id, count in cat_counts.most_common(15):
        cat_name = categories.get(cat_id, f"unknown_{cat_id}")
        print(f"  {cat_id}: {cat_name[:50]} ({count} annotations)")
    
    # Store section analysis from filenames
    print("\n=== Store Section Analysis ===")
    store_sections = defaultdict(list)
    section_patterns = {
        'Egg': ['egg'],
        'Frokost': ['frokost'],
        'Knekkebrod': ['knekkebrod'],
        'Varmedrikker': ['varmedrikker']
    }
    
    for img in coco_data['images']:
        fname = img['file_name'].lower()
        assigned = False
        for section, patterns in section_patterns.items():
            if any(pattern in fname for pattern in patterns):
                store_sections[section].append(img['id'])
                assigned = True
                break
        if not assigned:
            store_sections['unknown'].append(img['id'])
    
    print(f"Store section distribution:")
    for section, img_ids in store_sections.items():
        print(f"  {section}: {len(img_ids)} images")
    
    # Metadata analysis
    print("\n=== Metadata Analysis ===")
    metadata_path = data_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        print(f"Products in metadata: {len(metadata)}")
        
        # Check metadata structure
        sample_key = list(metadata.keys())[0]
        sample_value = metadata[sample_key]
        print(f"\nSample metadata entry:")
        print(f"  Key: {sample_key}")
        print(f"  Value: {sample_value}")
        print(f"  Value type: {type(sample_value)}")
        
        if isinstance(sample_value, dict):
            print(f"  Fields: {list(sample_value.keys())}")
        
        # Annotation count distribution from metadata
        if isinstance(sample_value, dict) and 'annotation_count' in sample_value:
            ann_counts_meta = [info['annotation_count'] for info in metadata.values() if isinstance(info, dict)]
            print(f"\nAnnotation counts from metadata:")
            print(f"  Products with 0 annotations: {sum(1 for c in ann_counts_meta if c == 0)}")
            print(f"  Products with 1 annotation: {sum(1 for c in ann_counts_meta if c == 1)}")
            print(f"  Products with 2-5 annotations: {sum(1 for c in ann_counts_meta if 2 <= c <= 5)}")
            print(f"  Products with 6+ annotations: {sum(1 for c in ann_counts_meta if c >= 6)}")
    
    # Products directory analysis
    print("\n=== Products Directory Analysis ===")
    products_path = data_path / "products"
    if products_path.exists():
        product_dirs = [d for d in products_path.iterdir() if d.is_dir()]
        print(f"Product directories: {len(product_dirs)}")
        
        # Image type analysis
        image_types = defaultdict(int)
        total_images = 0
        products_with_images = 0
        
        for product_dir in product_dirs[:100]:  # Sample first 100 to avoid too much output
            images = list(product_dir.glob("*.jpg"))
            if images:
                products_with_images += 1
                total_images += len(images)
                for img in images:
                    image_types[img.name] += 1
        
        print(f"\nReference image analysis (first 100 products):")
        print(f"  Products with reference images: {products_with_images}/100")
        print(f"  Total reference images: {total_images}")
        print(f"  Average images per product: {total_images/products_with_images:.1f}")
        
        print(f"\nImage type frequency:")
        for img_type, count in sorted(image_types.items()):
            print(f"  {img_type}: {count}")
        
        # Sample a few products in detail
        print(f"\nSample product details:")
        for i, product_dir in enumerate(product_dirs[:5]):
            images = list(product_dir.glob("*.jpg"))
            total_size = sum(img.stat().st_size for img in images)
            print(f"  {product_dir.name}: {len(images)} images, {total_size/1024:.1f} KB total")
            for img in images:
                size_kb = img.stat().st_size / 1024
                print(f"    {img.name}: {size_kb:.1f} KB")
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()