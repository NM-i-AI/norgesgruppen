import json
import pathlib
from collections import Counter
import numpy as np

def main():
    print("=== NorgesGruppen Dataset Structure and Statistics ===")
    
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
        return
    
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
        print(f"Average annotations per image: {len(annotations)/len(images):.1f}")
        
        # Annotations per image distribution
        print("\n--- Annotations per Image Distribution ---")
        img_ann_counts = Counter(ann['image_id'] for ann in annotations)
        ann_counts = sorted(img_ann_counts.values())
        print(f"Min annotations per image: {min(ann_counts)}")
        print(f"Max annotations per image: {max(ann_counts)}")
        print(f"Mean annotations per image: {sum(ann_counts)/len(ann_counts):.1f}")
        print(f"Median annotations per image: {ann_counts[len(ann_counts)//2]}")
        print(f"25th percentile: {ann_counts[len(ann_counts)//4]}")
        print(f"75th percentile: {ann_counts[3*len(ann_counts)//4]}")
        
        # Category frequency distribution
        print("\n--- Category Frequency Distribution ---")
        cat_counts = Counter(ann['category_id'] for ann in annotations)
        print(f"Categories with annotations: {len(cat_counts)}/{len(categories)}")
        
        print("\nTop 10 most frequent categories:")
        for cat_id, count in cat_counts.most_common(10):
            cat_name = next((c['name'] for c in categories if c['id'] == cat_id), 'Unknown')
            print(f"  {cat_id:3d}: {count:4d}x  {cat_name[:60]}")
        
        print("\nBottom 10 least frequent categories:")
        for cat_id, count in cat_counts.most_common()[-10:]:
            cat_name = next((c['name'] for c in categories if c['id'] == cat_id), 'Unknown')
            print(f"  {cat_id:3d}: {count:4d}x  {cat_name[:60]}")
        
        # Class imbalance analysis
        print("\n--- Class Imbalance Analysis ---")
        freq_counts = Counter(cat_counts.values())
        print(f"Categories with exactly 1 annotation: {sum(1 for c in cat_counts.values() if c == 1)}")
        print(f"Categories with <5 annotations: {sum(1 for c in cat_counts.values() if c < 5)}")
        print(f"Categories with <10 annotations: {sum(1 for c in cat_counts.values() if c < 10)}")
        print(f"Categories with <20 annotations: {sum(1 for c in cat_counts.values() if c < 20)}")
        print(f"Categories with >50 annotations: {sum(1 for c in cat_counts.values() if c > 50)}")
        print(f"Categories with >100 annotations: {sum(1 for c in cat_counts.values() if c > 100)}")
        
        # Check for unknown_product category (356)
        unknown_count = cat_counts.get(356, 0)
        print(f"\nCategory 356 'unknown_product' frequency: {unknown_count}")
        
        # Image sizes
        print("\n--- Image Sizes ---")
        size_counts = Counter((img['width'], img['height']) for img in images)
        for (w, h), count in sorted(size_counts.items()):
            print(f"  {w}x{h}: {count} images")
        
        # Bounding box analysis
        print("\n--- Bounding Box Analysis ---")
        bbox_areas = [ann.get('area', ann['bbox'][2] * ann['bbox'][3]) for ann in annotations]
        bbox_widths = [ann['bbox'][2] for ann in annotations]
        bbox_heights = [ann['bbox'][3] for ann in annotations]
        bbox_x = [ann['bbox'][0] for ann in annotations]
        bbox_y = [ann['bbox'][1] for ann in annotations]
        
        # Sort for percentile calculations
        bbox_areas.sort()
        bbox_widths.sort()
        bbox_heights.sort()
        
        print(f"Bbox areas - Min: {min(bbox_areas):.0f}, Max: {max(bbox_areas):.0f}, Median: {bbox_areas[len(bbox_areas)//2]:.0f}")
        print(f"Bbox widths - Min: {min(bbox_widths):.0f}, Max: {max(bbox_widths):.0f}, Median: {bbox_widths[len(bbox_widths)//2]:.0f}")
        print(f"Bbox heights - Min: {min(bbox_heights):.0f}, Max: {max(bbox_heights):.0f}, Median: {bbox_heights[len(bbox_heights)//2]:.0f}")
        
        # Small/large bbox analysis
        small_boxes = sum(1 for area in bbox_areas if area < 1000)
        large_boxes = sum(1 for area in bbox_areas if area > 50000)
        print(f"Small boxes (<1000 px²): {small_boxes} ({small_boxes/len(bbox_areas)*100:.1f}%)")
        print(f"Large boxes (>50000 px²): {large_boxes} ({large_boxes/len(bbox_areas)*100:.1f}%)")
        
        # Annotation fields analysis
        print("\n--- Annotation Fields ---")
        sample_ann = annotations[0]
        print(f"Sample annotation keys: {list(sample_ann.keys())}")
        print(f"Sample annotation: {sample_ann}")
        
        # Count corrected annotations
        corrected_count = sum(1 for ann in annotations if ann.get('corrected', False))
        print(f"\nCorrected annotations: {corrected_count}/{len(annotations)} ({corrected_count/len(annotations)*100:.1f}%)")
        
        # Product codes analysis
        product_codes = [ann.get('product_code') for ann in annotations if ann.get('product_code')]
        unique_product_codes = set(product_codes)
        print(f"Annotations with product codes: {len(product_codes)}/{len(annotations)}")
        print(f"Unique product codes in annotations: {len(unique_product_codes)}")
        
        # Check for missing product codes
        missing_codes = sum(1 for ann in annotations if not ann.get('product_code'))
        print(f"Annotations missing product codes: {missing_codes}")
        
    except Exception as e:
        print(f"Error loading annotations: {e}")
        import traceback
        traceback.print_exc()
    
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
        angle_counts = Counter()
        for p in products:
            angles = p.get('image_types', [])
            all_angles.update(angles)
            for angle in angles:
                angle_counts[angle] += 1
        
        print(f"\nAvailable image angles: {sorted(all_angles)}")
        print("Angle frequency:")
        for angle, count in sorted(angle_counts.items()):
            print(f"  {angle}: {count} products")
        
        # Annotation count distribution in metadata
        ann_counts = [p.get('annotation_count', 0) for p in products]
        ann_counts.sort()
        print(f"\nAnnotation counts per product (from metadata):")
        print(f"  Min: {min(ann_counts)}, Max: {max(ann_counts)}, Median: {ann_counts[len(ann_counts)//2]}")
        print(f"  Products with 0 annotations: {sum(1 for c in ann_counts if c == 0)}")
        print(f"  Products with 1 annotation: {sum(1 for c in ann_counts if c == 1)}")
        print(f"  Products with 2-5 annotations: {sum(1 for c in ann_counts if 2 <= c <= 5)}")
        print(f"  Products with 6-10 annotations: {sum(1 for c in ann_counts if 6 <= c <= 10)}")
        print(f"  Products with >10 annotations: {sum(1 for c in ann_counts if c > 10)}")
        print(f"  Products with >50 annotations: {sum(1 for c in ann_counts if c > 50)}")
        
        # Corrected count analysis
        corrected_counts = [p.get('corrected_count', 0) for p in products]
        total_corrected = sum(corrected_counts)
        print(f"\nTotal corrected annotations (from metadata): {total_corrected}")
        
        # Sample product details
        print("\nSample product entries:")
        for i, product in enumerate(products[:3]):
            print(f"  Product {i+1}: {product}")
        
    except Exception as e:
        print(f"Error loading metadata: {e}")
        import traceback
        traceback.print_exc()
    
    # Check products directory structure
    print("\n--- Products Directory Analysis ---")
    products_path = pathlib.Path('data/products')
    if products_path.exists():
        product_dirs = [d for d in products_path.iterdir() if d.is_dir()]
        print(f"Product directories: {len(product_dirs)}")
        
        if product_dirs:
            # Sample a few product directories
            sample_dir = product_dirs[0]
            image_files = list(sample_dir.glob('*.jpg'))
            print(f"\nSample product {sample_dir.name}:")
            print(f"  Images: {[f.name for f in image_files]}")
            
            # Count images per angle across all products
            angle_image_counts = Counter()
            total_product_images = 0
            
            for prod_dir in product_dirs:
                images_in_dir = list(prod_dir.glob('*.jpg'))
                total_product_images += len(images_in_dir)
                for img_file in images_in_dir:
                    angle = img_file.stem  # filename without extension
                    angle_image_counts[angle] += 1
            
            print(f"\nTotal product reference images: {total_product_images}")
            print(f"Images per angle:")
            for angle, count in sorted(angle_image_counts.items()):
                print(f"  {angle}: {count} images")
            
            # Check for products with missing main.jpg
            products_with_main = sum(1 for d in product_dirs if (d / 'main.jpg').exists())
            print(f"\nProducts with main.jpg: {products_with_main}/{len(product_dirs)}")
            
    else:
        print("data/products/ directory not found")
    
    # Cross-reference product codes between annotations and products directory
    print("\n--- Product Code Cross-Reference ---")
    try:
        # Get product codes from annotations
        ann_product_codes = set(ann.get('product_code') for ann in annotations if ann.get('product_code'))
        ann_product_codes.discard(None)  # Remove None values
        
        # Get product codes from products directory
        products_path = pathlib.Path('data/products')
        if products_path.exists():
            dir_product_codes = set(d.name for d in products_path.iterdir() if d.is_dir())
        else:
            dir_product_codes = set()
        
        # Get product codes from metadata
        metadata_product_codes = set(p.get('product_code') for p in products if p.get('product_code'))
        metadata_product_codes.discard(None)
        
        print(f"Product codes in annotations: {len(ann_product_codes)}")
        print(f"Product codes in metadata: {len(metadata_product_codes)}")
        print(f"Product directories: {len(dir_product_codes)}")
        
        # Find overlaps
        ann_with_dirs = ann_product_codes & dir_product_codes
        ann_without_dirs = ann_product_codes - dir_product_codes
        dirs_without_ann = dir_product_codes - ann_product_codes
        
        print(f"\nAnnotated products with reference images: {len(ann_with_dirs)}/{len(ann_product_codes)} ({len(ann_with_dirs)/len(ann_product_codes)*100:.1f}%)")
        print(f"Annotated products missing reference images: {len(ann_without_dirs)}")
        print(f"Reference images without annotations: {len(dirs_without_ann)}")
        
        if ann_without_dirs:
            print(f"\nSample annotated products missing reference images:")
            for code in sorted(list(ann_without_dirs))[:5]:
                print(f"  {code}")
        
        if dirs_without_ann:
            print(f"\nSample reference images without annotations:")
            for code in sorted(list(dirs_without_ann))[:5]:
                print(f"  {code}")
        
    except Exception as e:
        print(f"Error in product code cross-reference: {e}")
        import traceback
        traceback.print_exc()
    
    # Verify sample file paths
    print("\n--- File Path Verification ---")
    try:
        # Check sample train images
        train_images_path = pathlib.Path('data/train/images')
        if train_images_path.exists():
            train_images = list(train_images_path.glob('*.jpg'))[:5]
            print(f"Sample train images exist:")
            for img_path in train_images:
                exists = img_path.exists()
                size = img_path.stat().st_size if exists else 0
                print(f"  {img_path.name}: {exists} ({size} bytes)")
        
        # Check sample product reference images
        if product_dirs:
            sample_products = product_dirs[:3]
            print(f"\nSample product reference images exist:")
            for prod_dir in sample_products:
                print(f"  {prod_dir.name}/:")
                for img_file in prod_dir.glob('*.jpg'):
                    size = img_file.stat().st_size
                    print(f"    {img_file.name}: {size} bytes")
        
    except Exception as e:
        print(f"Error in file path verification: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n=== DATASET SUMMARY ===")
    try:
        print(f"• {len(images)} training images with {len(annotations)} annotations")
        print(f"• {len(categories)} categories, {len(cat_counts)} have annotations")
        print(f"• Severe class imbalance: {sum(1 for c in cat_counts.values() if c < 5)} categories have <5 examples")
        print(f"• {with_images} products have reference images ({len(product_dirs)} directories found)")
        print(f"• Average {len(annotations)/len(images):.1f} annotations per image")
        print(f"• {corrected_count/len(annotations)*100:.1f}% of annotations are manually corrected")
        print(f"• Detection challenge: dense shelves with {min(ann_counts)}-{max(ann_counts)} products per image")
        print(f"• Classification challenge: 357 categories, many with very few examples")
        print(f"• Product code linkage: {len(ann_with_dirs)}/{len(ann_product_codes)} annotated products have reference images")
    except:
        print("Error generating summary - some variables not defined")

if __name__ == "__main__":
    main()