import json
from pathlib import Path
from collections import Counter

def main():
    print("=== NorgesGruppen Dataset Exploration ===")
    
    # 1. Load annotations.json and examine structure
    print("\n1. Loading annotations.json...")
    annotations_path = Path("data/train/annotations.json")
    if not annotations_path.exists():
        print(f"ERROR: {annotations_path} not found")
        return
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    print(f"Annotation file structure:")
    print(f"  - Keys: {list(data.keys())}")
    print(f"  - Images: {len(data['images'])}")
    print(f"  - Categories: {len(data['categories'])}")
    print(f"  - Annotations: {len(data['annotations'])}")
    
    # 2. Count images, annotations, categories
    print("\n2. Basic counts:")
    num_images = len(data['images'])
    num_annotations = len(data['annotations'])
    num_categories = len(data['categories'])
    print(f"  - Total images: {num_images}")
    print(f"  - Total annotations: {num_annotations}")
    print(f"  - Total categories: {num_categories}")
    print(f"  - Average annotations per image: {num_annotations/num_images:.1f}")
    
    # 3. Annotations-per-image distribution
    print("\n3. Annotations per image distribution:")
    image_annotation_counts = Counter(ann['image_id'] for ann in data['annotations'])
    counts = sorted(image_annotation_counts.values())
    print(f"  - Min annotations per image: {min(counts)}")
    print(f"  - Max annotations per image: {max(counts)}")
    print(f"  - Median: {counts[len(counts)//2]}")
    print(f"  - Mean: {sum(counts)/len(counts):.1f}")
    print(f"  - 25th percentile: {counts[len(counts)//4]}")
    print(f"  - 75th percentile: {counts[3*len(counts)//4]}")
    
    # 4. Category frequency distribution
    print("\n4. Category frequency distribution:")
    category_counts = Counter(ann['category_id'] for ann in data['annotations'])
    
    # Count categories by frequency buckets
    freq_buckets = {
        '<5': sum(1 for count in category_counts.values() if count < 5),
        '<10': sum(1 for count in category_counts.values() if count < 10),
        '<20': sum(1 for count in category_counts.values() if count < 20),
        '<50': sum(1 for count in category_counts.values() if count < 50),
        '>=50': sum(1 for count in category_counts.values() if count >= 50)
    }
    
    print(f"  - Categories with <5 annotations: {freq_buckets['<5']}")
    print(f"  - Categories with <10 annotations: {freq_buckets['<10']}")
    print(f"  - Categories with <20 annotations: {freq_buckets['<20']}")
    print(f"  - Categories with <50 annotations: {freq_buckets['<50']}")
    print(f"  - Categories with >=50 annotations: {freq_buckets['>=50']}")
    
    print(f"\n  Top 10 most frequent categories:")
    for cat_id, count in category_counts.most_common(10):
        cat_name = next(c['name'] for c in data['categories'] if c['id'] == cat_id)
        print(f"    {cat_id:3d}: {count:4d}x  {cat_name[:50]}...")
    
    print(f"\n  Bottom 10 least frequent categories:")
    for cat_id, count in list(category_counts.most_common())[-10:]:
        cat_name = next(c['name'] for c in data['categories'] if c['id'] == cat_id)
        print(f"    {cat_id:3d}: {count:4d}x  {cat_name[:50]}...")
    
    # 5. Check image sizes
    print("\n5. Image sizes:")
    image_sizes = Counter((img['width'], img['height']) for img in data['images'])
    for (width, height), count in sorted(image_sizes.items()):
        print(f"  - {width}x{height}: {count} images")
    
    # 6. Examine product reference images structure
    print("\n6. Product reference images:")
    metadata_path = Path("data/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        products = metadata['products']
        products_with_images = [p for p in products if p['has_images']]
        
        print(f"  - Total products in metadata: {len(products)}")
        print(f"  - Products with reference images: {len(products_with_images)}")
        
        # Count available angles
        all_angles = set()
        for product in products_with_images:
            all_angles.update(product.get('image_types', []))
        print(f"  - Available image angles: {sorted(all_angles)}")
        
        # Angle frequency
        angle_counts = Counter()
        for product in products_with_images:
            for angle in product.get('image_types', []):
                angle_counts[angle] += 1
        
        print(f"  - Angle frequency:")
        for angle, count in sorted(angle_counts.items()):
            print(f"    {angle}: {count} products")
    else:
        print(f"  - Metadata file not found at {metadata_path}")
    
    # Check actual product folders
    products_dir = Path("data/products")
    if products_dir.exists():
        product_folders = [d.name for d in products_dir.iterdir() if d.is_dir()]
        print(f"  - Actual product folders found: {len(product_folders)}")
    else:
        print(f"  - Products directory not found at {products_dir}")
    
    # 7. Check overlap between annotation product_codes and available product image folders
    print("\n7. Product code overlap analysis:")
    
    # Get all product codes from annotations
    annotation_product_codes = set()
    for ann in data['annotations']:
        if 'product_code' in ann and ann['product_code']:
            annotation_product_codes.add(ann['product_code'])
    
    print(f"  - Unique product codes in annotations: {len(annotation_product_codes)}")
    
    if products_dir.exists():
        available_product_codes = set(d.name for d in products_dir.iterdir() if d.is_dir())
        print(f"  - Available product image folders: {len(available_product_codes)}")
        
        overlap = annotation_product_codes.intersection(available_product_codes)
        print(f"  - Product codes with both annotations and images: {len(overlap)}")
        
        missing_images = annotation_product_codes - available_product_codes
        print(f"  - Product codes in annotations but missing images: {len(missing_images)}")
        
        unused_images = available_product_codes - annotation_product_codes
        print(f"  - Product image folders not used in annotations: {len(unused_images)}")
        
        if missing_images:
            print(f"  - Examples of missing image product codes: {list(missing_images)[:5]}")
    
    # 8. Check corrected annotations
    print("\n8. Annotation correction status:")
    corrected_counts = Counter(ann.get('corrected', False) for ann in data['annotations'])
    
    print(f"  - Annotations with corrected=True: {corrected_counts.get(True, 0)}")
    print(f"  - Annotations with corrected=False: {corrected_counts.get(False, 0)}")
    print(f"  - Annotations missing 'corrected' field: {num_annotations - sum(corrected_counts.values())}")
    
    if corrected_counts.get(True, 0) > 0:
        correction_rate = corrected_counts.get(True, 0) / num_annotations * 100
        print(f"  - Correction rate: {correction_rate:.1f}%")
    
    # Additional useful stats
    print("\n9. Additional statistics:")
    
    # Bbox size analysis
    bbox_areas = [ann['area'] for ann in data['annotations'] if 'area' in ann]
    if bbox_areas:
        bbox_areas.sort()
        print(f"  - Bbox areas - Min: {min(bbox_areas):.0f}, Max: {max(bbox_areas):.0f}, Median: {bbox_areas[len(bbox_areas)//2]:.0f}")
    
    # Check for unknown_product category
    unknown_cats = [c for c in data['categories'] if 'unknown' in c['name'].lower()]
    if unknown_cats:
        for cat in unknown_cats:
            unknown_count = category_counts.get(cat['id'], 0)
            print(f"  - Unknown product category {cat['id']} ('{cat['name']}'): {unknown_count} annotations")
    
    print("\n=== Exploration Complete ===")

if __name__ == "__main__":
    main()