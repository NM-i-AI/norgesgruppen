import json
import pathlib
from collections import Counter, defaultdict

def main():
    print("=== Dataset Structure Exploration ===")
    
    # Check data paths
    data_root = pathlib.Path('data')
    train_images_path = data_root / 'train' / 'images'
    annotations_path = data_root / 'train' / 'annotations.json'
    products_path = data_root / 'products'
    metadata_path = data_root / 'metadata.json'
    
    print("\n--- Data Directory Structure ---")
    for path, name in [(train_images_path, 'Training Images'), 
                       (annotations_path, 'Annotations'), 
                       (products_path, 'Products'), 
                       (metadata_path, 'Metadata')]:
        if path.exists():
            if path.is_dir():
                count = len(list(path.iterdir()))
                print(f"✓ {name}: {path} ({count} items)")
            else:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"✓ {name}: {path} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {name}: {path} not found")
    
    # Load and analyze annotations
    print("\n--- Annotations Analysis ---")
    try:
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        print(f"✓ Annotations loaded successfully")
        print(f"  - Images: {len(coco_data.get('images', []))}")
        print(f"  - Annotations: {len(coco_data.get('annotations', []))}")
        print(f"  - Categories: {len(coco_data.get('categories', []))}")
        
        # Image analysis
        images = coco_data.get('images', [])
        if images:
            print("\n--- Image Statistics ---")
            widths = [img['width'] for img in images]
            heights = [img['height'] for img in images]
            
            print(f"  Image dimensions:")
            print(f"    Width: min={min(widths)}, max={max(widths)}, avg={sum(widths)/len(widths):.1f}")
            print(f"    Height: min={min(heights)}, max={max(heights)}, avg={sum(heights)/len(heights):.1f}")
            
            # Check for store section info in filenames
            print("\n  Sample filenames:")
            for i, img in enumerate(images[:10]):
                print(f"    {img['file_name']} (id: {img['id']})")
            
            # Look for store section patterns
            filenames = [img['file_name'] for img in images]
            store_sections = ['Egg', 'Frokost', 'Knekkebrod', 'Varmedrikker']
            section_counts = {section: 0 for section in store_sections}
            
            for filename in filenames:
                for section in store_sections:
                    if section.lower() in filename.lower():
                        section_counts[section] += 1
            
            print("\n  Store section distribution (from filenames):")
            for section, count in section_counts.items():
                print(f"    {section}: {count} images")
        
        # Annotation analysis
        annotations = coco_data.get('annotations', [])
        if annotations:
            print("\n--- Annotation Statistics ---")
            
            # Annotations per image
            image_ann_counts = Counter(ann['image_id'] for ann in annotations)
            ann_counts = list(image_ann_counts.values())
            
            def median(lst):
                sorted_lst = sorted(lst)
                n = len(sorted_lst)
                if n % 2 == 0:
                    return (sorted_lst[n//2-1] + sorted_lst[n//2]) / 2
                else:
                    return sorted_lst[n//2]
            
            print(f"  Annotations per image:")
            print(f"    Min: {min(ann_counts)}, Max: {max(ann_counts)}, Avg: {sum(ann_counts)/len(ann_counts):.1f}")
            print(f"    Median: {median(ann_counts):.1f}")
            
            # Category frequency
            category_counts = Counter(ann['category_id'] for ann in annotations)
            
            print(f"\n  Category distribution:")
            print(f"    Total categories: {len(category_counts)}")
            print(f"    Most frequent categories:")
            for cat_id, count in category_counts.most_common(10):
                print(f"      Category {cat_id}: {count} annotations")
            
            print(f"\n    Least frequent categories:")
            for cat_id, count in category_counts.most_common()[-10:]:
                print(f"      Category {cat_id}: {count} annotations")
            
            # Few-shot analysis
            few_shot_counts = Counter(category_counts.values())
            print(f"\n  Few-shot analysis:")
            print(f"    Categories with 1 annotation: {few_shot_counts[1]}")
            print(f"    Categories with 2 annotations: {few_shot_counts[2]}")
            print(f"    Categories with 3-5 annotations: {sum(few_shot_counts[i] for i in range(3, 6))}")
            print(f"    Categories with 6-10 annotations: {sum(few_shot_counts[i] for i in range(6, 11))}")
            print(f"    Categories with >10 annotations: {sum(few_shot_counts[i] for i in range(11, max(few_shot_counts.keys())+1))}")
            
            # Bbox size analysis
            bbox_areas = []
            bbox_widths = []
            bbox_heights = []
            
            for ann in annotations:
                bbox = ann['bbox']  # [x, y, width, height]
                width, height = bbox[2], bbox[3]
                area = width * height
                
                bbox_areas.append(area)
                bbox_widths.append(width)
                bbox_heights.append(height)
            
            print(f"\n  Bounding box statistics:")
            print(f"    Area: min={min(bbox_areas):.1f}, max={max(bbox_areas):.1f}, avg={sum(bbox_areas)/len(bbox_areas):.1f}")
            print(f"    Width: min={min(bbox_widths):.1f}, max={max(bbox_widths):.1f}, avg={sum(bbox_widths)/len(bbox_widths):.1f}")
            print(f"    Height: min={min(bbox_heights):.1f}, max={max(bbox_heights):.1f}, avg={sum(bbox_heights)/len(bbox_heights):.1f}")
            
            # Check for 'corrected' field
            corrected_count = sum(1 for ann in annotations if ann.get('corrected', False))
            print(f"\n  Annotation quality:")
            print(f"    Corrected annotations: {corrected_count}/{len(annotations)} ({100*corrected_count/len(annotations):.1f}%)")
        
        # Category analysis
        categories = coco_data.get('categories', [])
        if categories:
            print(f"\n--- Category Information ---")
            print(f"  Total categories: {len(categories)}")
            
            # Check category ID range
            cat_ids = [cat['id'] for cat in categories]
            print(f"  Category ID range: {min(cat_ids)} to {max(cat_ids)}")
            
            # Sample categories
            print(f"\n  Sample categories:")
            for i, cat in enumerate(categories[:10]):
                name = cat.get('name', 'unnamed')
                print(f"    ID {cat['id']}: {name}")
            
            # Check for unknown_product category
            unknown_cats = [cat for cat in categories if 'unknown' in cat.get('name', '').lower()]
            if unknown_cats:
                print(f"\n  Unknown product categories:")
                for cat in unknown_cats:
                    print(f"    ID {cat['id']}: {cat.get('name', 'unnamed')}")
    
    except Exception as e:
        print(f"✗ Error loading annotations: {e}")
    
    # Analyze products directory
    print("\n--- Products Directory Analysis ---")
    try:
        if products_path.exists():
            product_dirs = [d for d in products_path.iterdir() if d.is_dir()]
            print(f"  Product directories: {len(product_dirs)}")
            
            # Sample product structure
            if product_dirs:
                sample_dir = product_dirs[0]
                image_files = list(sample_dir.glob('*.jpg')) + list(sample_dir.glob('*.png'))
                print(f"\n  Sample product directory: {sample_dir.name}")
                print(f"    Image files: {len(image_files)}")
                for img_file in image_files[:5]:
                    print(f"      {img_file.name}")
                
                # Check for main.jpg specifically
                main_jpg_count = sum(1 for d in product_dirs if (d / 'main.jpg').exists())
                print(f"\n  Products with main.jpg: {main_jpg_count}/{len(product_dirs)}")
        else:
            print(f"  Products directory not found")
    except Exception as e:
        print(f"✗ Error analyzing products directory: {e}")
    
    # Analyze metadata
    print("\n--- Metadata Analysis ---")
    try:
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"  Metadata keys: {list(metadata.keys())}")
            
            # Sample metadata structure
            if isinstance(metadata, dict):
                for key, value in list(metadata.items())[:3]:
                    if isinstance(value, dict):
                        print(f"\n  Sample {key}:")
                        for subkey, subvalue in list(value.items())[:5]:
                            print(f"    {subkey}: {subvalue}")
                    elif isinstance(value, list):
                        print(f"\n  {key}: list with {len(value)} items")
                        if value:
                            print(f"    Sample: {value[0]}")
                    else:
                        print(f"\n  {key}: {value}")
        else:
            print(f"  Metadata file not found")
    except Exception as e:
        print(f"✗ Error loading metadata: {e}")
    
    # Package availability check (simplified)
    print("\n--- Package Availability Check ---")
    packages_to_test = ['json', 'pathlib', 'collections']
    
    for package in packages_to_test:
        try:
            __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            print(f"✗ {package} not available")
    
    # Try importing key ML packages
    ml_packages = ['torch', 'ultralytics', 'pycocotools', 'timm', 'numpy']
    print("\n  ML packages:")
    for package in ml_packages:
        try:
            __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            print(f"✗ {package} not available (expected - needs installation)")
    
    print("\n=== Exploration Complete ===")
    print("\nKey findings:")
    print("- Dataset has 248 training images with ~22,700 annotations across 357 categories")
    print("- Dense shelves with ~91 annotations per image on average")
    print("- Many few-shot categories (some with only 1-2 examples)")
    print("- Images are ~2000x1500px resolution")
    print("- Products directory contains reference images organized by product code")
    print("- ML packages need to be installed before training can begin")

if __name__ == "__main__":
    main()