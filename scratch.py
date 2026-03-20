import json
from pathlib import Path
from collections import defaultdict, Counter

def get_image_dimensions(img_path):
    """Get image dimensions without PIL as fallback"""
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    except ImportError:
        # PIL not available, skip dimension check
        return None
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None

def main():
    print("=== Dataset Exploration ===")
    
    # 1. Count images in data/train/images/
    images_dir = Path("data/train/images")
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
        print(f"\n1. Images: {len(image_files)} total")
        
        # Sample a few image dimensions
        sample_dims = []
        for i, img_path in enumerate(image_files[:10]):  # Sample first 10
            dims = get_image_dimensions(img_path)
            if dims:
                sample_dims.append(dims)
        
        if sample_dims:
            widths = [d[0] for d in sample_dims]
            heights = [d[1] for d in sample_dims]
            print(f"   Sample dimensions (first {len(sample_dims)}): {sample_dims}")
            print(f"   Width range: {min(widths)}-{max(widths)}, Height range: {min(heights)}-{max(heights)}")
        else:
            print("   Could not read image dimensions (PIL not available)")
    else:
        print("\n1. Images directory not found!")
    
    # 2. Examine annotations.json structure
    annotations_path = Path("data/train/annotations.json")
    if annotations_path.exists():
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        print(f"\n2. Annotations structure:")
        print(f"   Keys: {list(coco_data.keys())}")
        
        # Categories
        categories = coco_data.get('categories', [])
        print(f"   Categories: {len(categories)} total")
        if categories:
            print(f"   Category ID range: {min(c['id'] for c in categories)} - {max(c['id'] for c in categories)}")
            print(f"   Sample categories: {categories[:5]}")
        
        # Images
        images = coco_data.get('images', [])
        print(f"   Images: {len(images)} total")
        if images:
            print(f"   Sample image entry: {images[0]}")
        
        # Annotations
        annotations = coco_data.get('annotations', [])
        print(f"   Annotations: {len(annotations)} total")
        if annotations:
            print(f"   Sample annotation: {annotations[0]}")
            
            # Annotations per image
            img_ann_count = defaultdict(int)
            for ann in annotations:
                img_ann_count[ann['image_id']] += 1
            
            ann_counts = list(img_ann_count.values())
            mean_ann = sum(ann_counts) / len(ann_counts) if ann_counts else 0
            print(f"   Annotations per image - Min: {min(ann_counts)}, Max: {max(ann_counts)}, Mean: {mean_ann:.1f}")
            
            # Category distribution
            cat_counts = Counter(ann['category_id'] for ann in annotations)
            print(f"   Categories with annotations: {len(cat_counts)}")
            print(f"   Most common categories: {cat_counts.most_common(10)}")
            print(f"   Least common categories: {cat_counts.most_common()[-10:]}")
            
            # Categories with very few annotations
            rare_cats = sum(1 for count in cat_counts.values() if count <= 5)
            print(f"   Categories with ≤5 annotations: {rare_cats}")
            
    else:
        print("\n2. Annotations file not found!")
    
    # 3. Examine metadata.json
    metadata_path = Path("data/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n3. Metadata structure:")
        print(f"   Keys: {list(metadata.keys())}")
        
        if 'products' in metadata:
            products = metadata['products']
            print(f"   Products: {len(products)} total")
            
            # Sample product entry - fix the bug here
            if isinstance(products, dict):
                sample_product = next(iter(products.values())) if products else None
            elif isinstance(products, list):
                sample_product = products[0] if products else None
            else:
                sample_product = None
                
            if sample_product:
                print(f"   Sample product: {sample_product}")
            
            # Count products with different image types
            image_types = defaultdict(int)
            if isinstance(products, dict):
                product_values = products.values()
            elif isinstance(products, list):
                product_values = products
            else:
                product_values = []
                
            for product in product_values:
                available_images = product.get('available_images', [])
                for img_type in available_images:
                    image_types[img_type] += 1
            
            print(f"   Image types available: {dict(image_types)}")
    else:
        print("\n3. Metadata file not found!")
    
    # 4. Examine reference product images
    products_dir = Path("data/products")
    if products_dir.exists():
        product_dirs = [d for d in products_dir.iterdir() if d.is_dir()]
        print(f"\n4. Reference product images:")
        print(f"   Product directories: {len(product_dirs)}")
        
        # Sample a few product directories
        sample_products = product_dirs[:5]
        for prod_dir in sample_products:
            image_files = list(prod_dir.glob("*.jpg")) + list(prod_dir.glob("*.jpeg")) + list(prod_dir.glob("*.png"))
            print(f"   {prod_dir.name}: {len(image_files)} images - {[f.name for f in image_files]}")
        
        # Count total reference images
        total_ref_images = 0
        for prod_dir in product_dirs:
            image_files = list(prod_dir.glob("*.jpg")) + list(prod_dir.glob("*.jpeg")) + list(prod_dir.glob("*.png"))
            total_ref_images += len(image_files)
        
        print(f"   Total reference images: {total_ref_images}")
    else:
        print("\n4. Products directory not found!")
    
    # 5. Check for store sections (if available in metadata)
    if annotations_path.exists():
        # Look for store section information in image filenames or metadata
        print(f"\n5. Store sections analysis:")
        
        # Try to extract store sections from image filenames
        section_counts = defaultdict(int)
        if images_dir.exists():
            for img_file in image_files:
                filename = img_file.stem.lower()
                # Look for section keywords
                if 'egg' in filename:
                    section_counts['Egg'] += 1
                elif 'frokost' in filename:
                    section_counts['Frokost'] += 1
                elif 'knekkebrod' in filename:
                    section_counts['Knekkebrod'] += 1
                elif 'varmedrikker' in filename:
                    section_counts['Varmedrikker'] += 1
                else:
                    section_counts['Unknown'] += 1
        
        print(f"   Store sections from filenames: {dict(section_counts)}")
    
    print("\n=== Exploration Complete ===")

if __name__ == "__main__":
    main()