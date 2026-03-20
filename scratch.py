import json
import pathlib
from collections import Counter

def main():
    print("=== Dataset Structure and Statistics ===")
    
    # Check file structure
    print("\n--- File Structure ---")
    data_path = pathlib.Path('data')
    if data_path.exists():
        print(f"data/ exists")
        for item in data_path.iterdir():
            if item.is_dir():
                print(f"  {item.name}/ (directory)")
                # Check subdirectories
                for subitem in item.iterdir():
                    if subitem.is_dir():
                        count = len(list(subitem.iterdir()))
                        print(f"    {subitem.name}/ ({count} items)")
                    else:
                        print(f"    {subitem.name} (file)")
            else:
                print(f"  {item.name} (file)")
    else:
        print("data/ directory not found")
    
    # Load annotations
    print("\n--- Annotations Analysis ---")
    try:
        with open('data/train/annotations.json', 'r') as f:
            annotations_data = json.load(f)
        
        images = annotations_data['images']
        annotations = annotations_data['annotations']
        categories = annotations_data['categories']
        
        print(f"Total images: {len(images)}")
        print(f"Total annotations: {len(annotations)}")
        print(f"Total categories: {len(categories)}")
        
        # Annotations per image distribution
        print("\n--- Annotations per Image ---")
        img_ann_counts = Counter(ann['image_id'] for ann in annotations)
        ann_counts = sorted(img_ann_counts.values())
        print(f"Min annotations per image: {min(ann_counts)}")
        print(f"Max annotations per image: {max(ann_counts)}")
        print(f"Mean annotations per image: {sum(ann_counts)/len(ann_counts):.1f}")
        print(f"Median annotations per image: {ann_counts[len(ann_counts)//2]}")
        
        # Category frequency distribution
        print("\n--- Category Frequency (Top 10) ---")
        cat_counts = Counter(ann['category_id'] for ann in annotations)
        for cat_id, count in cat_counts.most_common(10):
            cat_name = next(c['name'] for c in categories if c['id'] == cat_id)
            print(f"  {cat_id:3d}: {count:4d}x  {cat_name[:50]}")
        
        print(f"\nCategories with <5 annotations: {sum(1 for c in cat_counts.values() if c < 5)}")
        print(f"Categories with only 1 annotation: {sum(1 for c in cat_counts.values() if c == 1)}")
        
        # Image sizes
        print("\n--- Image Sizes ---")
        size_counts = Counter((img['width'], img['height']) for img in images)
        for (w, h), count in sorted(size_counts.items()):
            print(f"  {w}x{h}: {count} images")
        
        # Bbox size distribution
        print("\n--- Bounding Box Size Distribution ---")
        bbox_areas = [ann['area'] for ann in annotations if 'area' in ann]
        bbox_widths = [ann['bbox'][2] for ann in annotations]
        bbox_heights = [ann['bbox'][3] for ann in annotations]
        
        if bbox_areas:
            bbox_areas.sort()
            print(f"Bbox areas - Min: {min(bbox_areas):.0f}, Max: {max(bbox_areas):.0f}, Median: {bbox_areas[len(bbox_areas)//2]:.0f}")
        
        bbox_widths.sort()
        bbox_heights.sort()
        print(f"Bbox widths - Min: {min(bbox_widths):.0f}, Max: {max(bbox_widths):.0f}, Median: {bbox_widths[len(bbox_widths)//2]:.0f}")
        print(f"Bbox heights - Min: {min(bbox_heights):.0f}, Max: {max(bbox_heights):.0f}, Median: {bbox_heights[len(bbox_heights)//2]:.0f}")
        
        # Check for special fields
        print("\n--- Annotation Fields ---")
        sample_ann = annotations[0]
        print(f"Sample annotation keys: {list(sample_ann.keys())}")
        
        # Count corrected annotations
        corrected_count = sum(1 for ann in annotations if ann.get('corrected', False))
        print(f"Corrected annotations: {corrected_count}/{len(annotations)} ({corrected_count/len(annotations)*100:.1f}%)")
        
        # Product codes
        product_codes = set(ann.get('product_code') for ann in annotations if ann.get('product_code'))
        print(f"Unique product codes in annotations: {len(product_codes)}")
        
    except Exception as e:
        print(f"Error loading annotations: {e}")
    
    # Load metadata
    print("\n--- Metadata Analysis ---")
    try:
        with open('data/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        products = metadata['products']
        print(f"Total products in metadata: {len(products)}")
        
        # Products with images
        with_images = sum(1 for p in products if p.get('has_images', False))
        print(f"Products with reference images: {with_images}/{len(products)} ({with_images/len(products)*100:.1f}%)")
        
        # Available image angles
        all_angles = set()
        for p in products:
            all_angles.update(p.get('image_types', []))
        print(f"Available image angles: {sorted(all_angles)}")
        
        # Annotation count distribution
        ann_counts = [p.get('annotation_count', 0) for p in products]
        ann_counts.sort()
        print(f"\nAnnotation counts per product:")
        print(f"  Min: {min(ann_counts)}, Max: {max(ann_counts)}, Median: {ann_counts[len(ann_counts)//2]}")
        print(f"  Products with 0 annotations: {sum(1 for c in ann_counts if c == 0)}")
        print(f"  Products with 1 annotation: {sum(1 for c in ann_counts if c == 1)}")
        print(f"  Products with >10 annotations: {sum(1 for c in ann_counts if c > 10)}")
        
    except Exception as e:
        print(f"Error loading metadata: {e}")
    
    # Check products directory structure
    print("\n--- Products Directory ---")
    products_path = pathlib.Path('data/products')
    if products_path.exists():
        product_dirs = [d for d in products_path.iterdir() if d.is_dir()]
        print(f"Product directories: {len(product_dirs)}")
        
        if product_dirs:
            # Sample a few product directories
            sample_dir = product_dirs[0]
            image_files = list(sample_dir.glob('*.jpg'))
            print(f"Sample product {sample_dir.name} has {len(image_files)} images: {[f.name for f in image_files]}")
            
            # Count total images across all products
            total_product_images = 0
            for prod_dir in product_dirs[:10]:  # Sample first 10 to avoid timeout
                total_product_images += len(list(prod_dir.glob('*.jpg')))
            print(f"Total product images (first 10 products): {total_product_images}")
    else:
        print("data/products/ directory not found")

if __name__ == "__main__":
    main()