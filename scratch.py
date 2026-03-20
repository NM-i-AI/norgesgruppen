import json
import os
from pathlib import Path
from collections import Counter, defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available, using basic statistics")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available, skipping image size analysis")

def mean(values):
    """Simple mean calculation without numpy"""
    return sum(values) / len(values) if values else 0

def main():
    print("=== NorgesGruppen Grocery Product Detection - Data Exploration ===")
    
    # 1. Examine image directory
    print("\n1. IMAGE DIRECTORY ANALYSIS")
    images_dir = Path("data/train/images")
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
        print(f"Total images: {len(image_files)}")
        
        # Sample a few images to check sizes
        if image_files and HAS_PIL:
            sizes = []
            for i, img_path in enumerate(image_files[:10]):  # Check first 10 images
                try:
                    with Image.open(img_path) as img:
                        sizes.append(img.size)  # (width, height)
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")
            
            if sizes:
                widths = [s[0] for s in sizes]
                heights = [s[1] for s in sizes]
                print(f"Sample image sizes (first 10):")
                print(f"  Width range: {min(widths)} - {max(widths)}")
                print(f"  Height range: {min(heights)} - {max(heights)}")
                if HAS_NUMPY:
                    print(f"  Average: {np.mean(widths):.0f} x {np.mean(heights):.0f}")
                else:
                    print(f"  Average: {mean(widths):.0f} x {mean(heights):.0f}")
        elif not HAS_PIL:
            print("  PIL not available, skipping image size analysis")
    else:
        print(f"Images directory not found: {images_dir}")
    
    # 2. Examine annotations.json
    print("\n2. ANNOTATIONS.JSON ANALYSIS")
    annotations_path = Path("data/train/annotations.json")
    images = None  # Initialize images variable
    if annotations_path.exists():
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        print(f"COCO format keys: {list(coco_data.keys())}")
        
        # Images
        images = coco_data.get('images', [])
        print(f"Number of images in annotations: {len(images)}")
        if images:
            print(f"Sample image entry: {images[0]}")
        
        # Categories
        categories = coco_data.get('categories', [])
        print(f"Number of categories: {len(categories)}")
        if categories:
            print(f"Category ID range: {min(c['id'] for c in categories)} - {max(c['id'] for c in categories)}")
            print(f"Sample categories:")
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
            print(f"Annotations with 'corrected=True': {corrected_count}")
            
            # Annotations per image
            image_ann_counts = Counter(ann['image_id'] for ann in annotations)
            ann_per_img = list(image_ann_counts.values())
            if HAS_NUMPY:
                print(f"Annotations per image - min: {min(ann_per_img)}, max: {max(ann_per_img)}, avg: {np.mean(ann_per_img):.1f}")
            else:
                print(f"Annotations per image - min: {min(ann_per_img)}, max: {max(ann_per_img)}, avg: {mean(ann_per_img):.1f}")
            
            # Annotations per category
            category_ann_counts = Counter(ann['category_id'] for ann in annotations)
            print(f"Categories with annotations: {len(category_ann_counts)}")
            ann_per_cat = list(category_ann_counts.values())
            if HAS_NUMPY:
                print(f"Annotations per category - min: {min(ann_per_cat)}, max: {max(ann_per_cat)}, avg: {np.mean(ann_per_cat):.1f}")
            else:
                print(f"Annotations per category - min: {min(ann_per_cat)}, max: {max(ann_per_cat)}, avg: {mean(ann_per_cat):.1f}")
            
            # Show most and least frequent categories
            most_common = category_ann_counts.most_common(5)
            least_common = category_ann_counts.most_common()[-5:]
            print(f"Most frequent categories: {most_common}")
            print(f"Least frequent categories: {least_common}")
            
            # Categories with very few examples (few-shot)
            few_shot_cats = [cat_id for cat_id, count in category_ann_counts.items() if count <= 5]
            print(f"Categories with ≤5 annotations: {len(few_shot_cats)} ({len(few_shot_cats)/len(category_ann_counts)*100:.1f}%)")
            
            # Bbox size analysis
            bbox_areas = []
            bbox_widths = []
            bbox_heights = []
            for ann in annotations[:1000]:  # Sample first 1000 for speed
                bbox = ann['bbox']  # [x, y, width, height]
                bbox_areas.append(bbox[2] * bbox[3])
                bbox_widths.append(bbox[2])
                bbox_heights.append(bbox[3])
            
            print(f"Bbox analysis (sample of 1000):")
            if HAS_NUMPY:
                print(f"  Area - min: {min(bbox_areas):.0f}, max: {max(bbox_areas):.0f}, avg: {np.mean(bbox_areas):.0f}")
                print(f"  Width - min: {min(bbox_widths):.0f}, max: {max(bbox_widths):.0f}, avg: {np.mean(bbox_widths):.0f}")
                print(f"  Height - min: {min(bbox_heights):.0f}, max: {max(bbox_heights):.0f}, avg: {np.mean(bbox_heights):.0f}")
            else:
                print(f"  Area - min: {min(bbox_areas):.0f}, max: {max(bbox_areas):.0f}, avg: {mean(bbox_areas):.0f}")
                print(f"  Width - min: {min(bbox_widths):.0f}, max: {max(bbox_widths):.0f}, avg: {mean(bbox_widths):.0f}")
                print(f"  Height - min: {min(bbox_heights):.0f}, max: {max(bbox_heights):.0f}, avg: {mean(bbox_heights):.0f}")
    else:
        print(f"Annotations file not found: {annotations_path}")
    
    # 3. Examine products directory
    print("\n3. PRODUCTS DIRECTORY ANALYSIS")
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
                images_in_dir = list(prod_dir.glob("*.jpg")) + list(prod_dir.glob("*.jpeg")) + list(prod_dir.glob("*.png"))
                total_ref_images += len(images_in_dir)
                for img in images_in_dir:
                    image_types[img.stem] += 1  # Count by filename (main, front, etc.)
            
            print(f"Total reference images: {total_ref_images}")
            print(f"Image types distribution: {dict(image_types.most_common(10))}")
    else:
        print(f"Products directory not found: {products_dir}")
    
    # 4. Examine metadata.json
    print("\n4. METADATA.JSON ANALYSIS")
    metadata_path = Path("data/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Metadata keys: {list(metadata.keys())}")
        
        if 'products' in metadata:
            products = metadata['products']
            print(f"Number of products in metadata: {len(products)}")
            
            # Sample product entry - handle both list and dict formats
            if products:
                if isinstance(products, list):
                    sample_product = products[0]
                    print(f"Sample product entry (first item): {sample_product}")
                    
                    # Check what fields are available
                    all_fields = set()
                    for prod_data in products:
                        if isinstance(prod_data, dict):
                            all_fields.update(prod_data.keys())
                    print(f"Available product fields: {sorted(all_fields)}")
                    
                    # Check annotation counts if available
                    if 'annotation_count' in all_fields:
                        ann_counts = [prod.get('annotation_count', 0) for prod in products if isinstance(prod, dict)]
                        if ann_counts:
                            if HAS_NUMPY:
                                print(f"Annotation counts in metadata - min: {min(ann_counts)}, max: {max(ann_counts)}, avg: {np.mean(ann_counts):.1f}")
                            else:
                                print(f"Annotation counts in metadata - min: {min(ann_counts)}, max: {max(ann_counts)}, avg: {mean(ann_counts):.1f}")
                    
                    # Check available image types
                    if 'available_images' in all_fields:
                        all_image_types = set()
                        for prod in products:
                            if isinstance(prod, dict) and 'available_images' in prod:
                                all_image_types.update(prod['available_images'])
                        print(f"Available image types: {sorted(all_image_types)}")
                        
                elif isinstance(products, dict):
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
                        if HAS_NUMPY:
                            print(f"Annotation counts in metadata - min: {min(ann_counts)}, max: {max(ann_counts)}, avg: {np.mean(ann_counts):.1f}")
                        else:
                            print(f"Annotation counts in metadata - min: {min(ann_counts)}, max: {max(ann_counts)}, avg: {mean(ann_counts):.1f}")
                    
                    # Check available image types
                    if 'available_images' in all_fields:
                        all_image_types = set()
                        for prod in products.values():
                            if 'available_images' in prod:
                                all_image_types.update(prod['available_images'])
                        print(f"Available image types: {sorted(all_image_types)}")
    else:
        print(f"Metadata file not found: {metadata_path}")
    
    # 5. Store sections analysis (if available in metadata)
    print("\n5. STORE SECTIONS ANALYSIS")
    if annotations_path.exists() and images is not None:
        # Try to find store section info in image filenames or metadata
        image_names = [img['file_name'] for img in images]
        if image_names:
            print(f"Sample image filenames: {image_names[:5]}")
            
            # Look for section patterns in filenames
            sections = ['Egg', 'Frokost', 'Knekkebrod', 'Varmedrikker']
            section_counts = {section: 0 for section in sections}
            
            for img_name in image_names:
                for section in sections:
                    if section.lower() in img_name.lower():
                        section_counts[section] += 1
                        break
            
            print(f"Images by store section (from filenames): {section_counts}")
            unassigned = len(image_names) - sum(section_counts.values())
            print(f"Images without clear section assignment: {unassigned}")
    
    print("\n=== Data Exploration Complete ===")

if __name__ == "__main__":
    main()