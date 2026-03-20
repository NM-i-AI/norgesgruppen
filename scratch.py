import json
import subprocess
import sys
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median

def main():
    print("=== NorgesGruppen Grocery Product Detection - Environment & Data Exploration ===")
    
    # 1. Check environment and installed packages
    print("\n1. ENVIRONMENT ANALYSIS")
    print(f"Python version: {sys.version}")
    
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("\nInstalled packages:")
            lines = result.stdout.strip().split('\n')
            for line in lines[:20]:  # Show first 20 packages
                print(f"  {line}")
            if len(lines) > 20:
                print(f"  ... and {len(lines) - 20} more packages")
        else:
            print(f"Error running pip list: {result.stderr}")
    except Exception as e:
        print(f"Could not run pip list: {e}")
    
    # Check specific packages
    print("\nChecking key packages:")
    packages_to_check = ['torch', 'torchvision', 'ultralytics', 'numpy', 'PIL', 'pycocotools']
    for pkg in packages_to_check:
        try:
            if pkg == 'PIL':
                import PIL
                print(f"  ✓ PIL (Pillow) available: {PIL.__version__}")
            else:
                module = __import__(pkg)
                version = getattr(module, '__version__', 'unknown')
                print(f"  ✓ {pkg} available: {version}")
        except ImportError:
            print(f"  ✗ {pkg} not available")
    
    # Check CUDA if torch is available
    try:
        import torch
        print(f"\nTorch CUDA info:")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name()}")
    except ImportError:
        print("\nTorch not available for CUDA check")
    
    # 2. Examine image directory
    print("\n2. IMAGE DIRECTORY ANALYSIS")
    images_dir = Path("data/train/images")
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
        print(f"Total images: {len(image_files)}")
        
        if image_files:
            print(f"Sample image filenames:")
            for i, img_path in enumerate(image_files[:10]):
                print(f"  {img_path.name}")
            
            # Try to get file sizes
            file_sizes = []
            for img_path in image_files[:20]:  # Check first 20
                try:
                    size = img_path.stat().st_size
                    file_sizes.append(size)
                except Exception as e:
                    print(f"Error getting size for {img_path}: {e}")
            
            if file_sizes:
                print(f"File sizes (first 20 images):")
                print(f"  Min: {min(file_sizes):,} bytes")
                print(f"  Max: {max(file_sizes):,} bytes")
                print(f"  Average: {mean(file_sizes):,.0f} bytes")
                print(f"  Median: {median(file_sizes):,.0f} bytes")
    else:
        print(f"Images directory not found: {images_dir}")
    
    # 3. Examine annotations.json
    print("\n3. ANNOTATIONS.JSON ANALYSIS")
    annotations_path = Path("data/train/annotations.json")
    if annotations_path.exists():
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        print(f"COCO format keys: {list(coco_data.keys())}")
        
        # Images
        images = coco_data.get('images', [])
        print(f"Number of images in annotations: {len(images)}")
        if images:
            print(f"Sample image entry: {images[0]}")
            
            # Check for store sections in filenames
            sections = ['Egg', 'Frokost', 'Knekkebrod', 'Varmedrikker']
            section_counts = {section: 0 for section in sections}
            unassigned = 0
            
            for img in images:
                filename = img.get('file_name', '').lower()
                assigned = False
                for section in sections:
                    if section.lower() in filename:
                        section_counts[section] += 1
                        assigned = True
                        break
                if not assigned:
                    unassigned += 1
            
            print(f"Images by store section (from filenames):")
            for section, count in section_counts.items():
                print(f"  {section}: {count}")
            print(f"  Unassigned: {unassigned}")
        
        # Categories
        categories = coco_data.get('categories', [])
        print(f"Number of categories: {len(categories)}")
        if categories:
            cat_ids = [c['id'] for c in categories]
            print(f"Category ID range: {min(cat_ids)} - {max(cat_ids)}")
            print(f"Sample categories (first 5):")
            for i, cat in enumerate(categories[:5]):
                print(f"  {cat}")
            print(f"Last few categories:")
            for cat in categories[-3:]:
                print(f"  {cat}")
        
        # Annotations
        annotations = coco_data.get('annotations', [])
        print(f"Number of annotations: {len(annotations)}")
        if annotations:
            print(f"Sample annotation: {annotations[0]}")
            
            # Check for 'corrected' field
            corrected_count = sum(1 for ann in annotations if ann.get('corrected', False))
            print(f"Annotations with 'corrected=True': {corrected_count} ({corrected_count/len(annotations)*100:.1f}%)")
            
            # Annotations per image
            image_ann_counts = Counter(ann['image_id'] for ann in annotations)
            ann_per_img = list(image_ann_counts.values())
            print(f"Annotations per image:")
            print(f"  Min: {min(ann_per_img)}, Max: {max(ann_per_img)}")
            print(f"  Average: {mean(ann_per_img):.1f}, Median: {median(ann_per_img):.1f}")
            
            # Annotations per category
            category_ann_counts = Counter(ann['category_id'] for ann in annotations)
            print(f"Categories with annotations: {len(category_ann_counts)}")
            ann_per_cat = list(category_ann_counts.values())
            print(f"Annotations per category:")
            print(f"  Min: {min(ann_per_cat)}, Max: {max(ann_per_cat)}")
            print(f"  Average: {mean(ann_per_cat):.1f}, Median: {median(ann_per_cat):.1f}")
            
            # Show most and least frequent categories
            most_common = category_ann_counts.most_common(5)
            least_common = sorted(category_ann_counts.items(), key=lambda x: x[1])[:5]
            print(f"Most frequent categories: {most_common}")
            print(f"Least frequent categories: {least_common}")
            
            # Categories with very few examples (few-shot)
            few_shot_thresholds = [1, 2, 5, 10]
            for threshold in few_shot_thresholds:
                few_shot_cats = [cat_id for cat_id, count in category_ann_counts.items() if count <= threshold]
                print(f"Categories with ≤{threshold} annotations: {len(few_shot_cats)} ({len(few_shot_cats)/len(category_ann_counts)*100:.1f}%)")
            
            # Bbox size analysis (sample for speed)
            sample_size = min(1000, len(annotations))
            bbox_areas = []
            bbox_widths = []
            bbox_heights = []
            
            for ann in annotations[:sample_size]:
                bbox = ann['bbox']  # [x, y, width, height]
                bbox_areas.append(bbox[2] * bbox[3])
                bbox_widths.append(bbox[2])
                bbox_heights.append(bbox[3])
            
            print(f"Bbox analysis (sample of {sample_size}):")
            print(f"  Area - Min: {min(bbox_areas):.0f}, Max: {max(bbox_areas):.0f}, Avg: {mean(bbox_areas):.0f}")
            print(f"  Width - Min: {min(bbox_widths):.0f}, Max: {max(bbox_widths):.0f}, Avg: {mean(bbox_widths):.0f}")
            print(f"  Height - Min: {min(bbox_heights):.0f}, Max: {max(bbox_heights):.0f}, Avg: {mean(bbox_heights):.0f}")
    else:
        print(f"Annotations file not found: {annotations_path}")
    
    # 4. Examine products directory
    print("\n4. PRODUCTS DIRECTORY ANALYSIS")
    products_dir = Path("data/products")
    if products_dir.exists():
        product_dirs = [d for d in products_dir.iterdir() if d.is_dir()]
        print(f"Number of product directories: {len(product_dirs)}")
        
        if product_dirs:
            # Sample a few product directories
            sample_products = product_dirs[:5]
            print(f"Sample product directories:")
            for prod_dir in sample_products:
                image_files = list(prod_dir.glob("*.jpg")) + list(prod_dir.glob("*.jpeg")) + list(prod_dir.glob("*.png"))
                print(f"  {prod_dir.name}: {len(image_files)} images")
                if image_files:
                    print(f"    Files: {[f.name for f in image_files]}")
            
            # Count total reference images
            total_ref_images = 0
            image_types = Counter()
            for prod_dir in product_dirs:
                images = list(prod_dir.glob("*.jpg")) + list(prod_dir.glob("*.jpeg")) + list(prod_dir.glob("*.png"))
                total_ref_images += len(images)
                for img in images:
                    image_types[img.stem] += 1  # Count by filename (main, front, etc.)
            
            print(f"Total reference images: {total_ref_images}")
            print(f"Image types distribution: {dict(image_types.most_common(10))}")
    else:
        print(f"Products directory not found: {products_dir}")
    
    # 5. Examine metadata.json
    print("\n5. METADATA.JSON ANALYSIS")
    metadata_path = Path("data/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Metadata keys: {list(metadata.keys())}")
        
        if 'products' in metadata:
            products = metadata['products']
            print(f"Number of products in metadata: {len(products)}")
            
            # Sample product entry
            if products:
                # Fix: products is a list, not a dict
                if isinstance(products, list):
                    sample_product = products[0]
                    print(f"Sample product entry: {sample_product}")
                    
                    # Check what fields are available
                    all_fields = set()
                    for prod_data in products:
                        all_fields.update(prod_data.keys())
                    print(f"Available product fields: {sorted(all_fields)}")
                    
                    # Check annotation counts if available
                    if 'annotation_count' in all_fields:
                        ann_counts = [prod.get('annotation_count', 0) for prod in products]
                        print(f"Annotation counts in metadata:")
                        print(f"  Min: {min(ann_counts)}, Max: {max(ann_counts)}, Avg: {mean(ann_counts):.1f}")
                    
                    # Check available image types
                    if 'available_images' in all_fields:
                        all_image_types = set()
                        for prod in products:
                            if 'available_images' in prod:
                                all_image_types.update(prod['available_images'])
                        print(f"Available image types: {sorted(all_image_types)}")
                else:
                    # If it's a dict (original assumption)
                    sample_key = list(products.keys())[0]
                    print(f"Sample product entry ({sample_key}): {products[sample_key]}")
                    
                    # Check what fields are available
                    all_fields = set()
                    for prod_data in products.values():
                        all_fields.update(prod_data.keys())
                    print(f"Available product fields: {sorted(all_fields)}")
                    
                    # Check annotation counts if available
                    if 'annotation_count' in all_fields:
                        ann_counts = [prod.get('annotation_count', 0) for prod in products.values()]
                        print(f"Annotation counts in metadata:")
                        print(f"  Min: {min(ann_counts)}, Max: {max(ann_counts)}, Avg: {mean(ann_counts):.1f}")
                    
                    # Check available image types
                    if 'available_images' in all_fields:
                        all_image_types = set()
                        for prod in products.values():
                            if 'available_images' in prod:
                                all_image_types.update(prod['available_images'])
                        print(f"Available image types: {sorted(all_image_types)}")
    else:
        print(f"Metadata file not found: {metadata_path}")
    
    print("\n=== Environment & Data Exploration Complete ===")
    
    # Summary for next steps
    print("\n6. SUMMARY FOR NEXT STEPS")
    if annotations_path.exists():
        print("✓ Dataset appears to be available")
        print("✓ COCO format annotations found")
        print("✓ Reference product images found")
        print("Next steps:")
        print("  1. Install required packages (ultralytics==8.1.0, torch, etc.)")
        print("  2. Create 90/10 train/val split")
        print("  3. Implement evaluation function")
        print("  4. Train baseline YOLOv8n model")
    else:
        print("✗ Dataset not found - check data paths")

if __name__ == "__main__":
    main()