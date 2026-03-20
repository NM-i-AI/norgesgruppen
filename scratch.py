import json
import os
from pathlib import Path
from collections import Counter, defaultdict

def explore_data():
    print("=== NorgesGruppen Grocery Dataset Exploration ===")
    
    # 1. Check images directory
    images_dir = Path("data/train/images")
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
        print(f"\n1. Images: {len(image_files)} files in {images_dir}")
        if image_files:
            print(f"   Sample files: {[f.name for f in image_files[:3]]}")
    else:
        print(f"\n1. Images directory not found: {images_dir}")
    
    # 2. Examine annotations.json
    annotations_path = Path("data/train/annotations.json")
    if annotations_path.exists():
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        print(f"\n2. Annotations structure:")
        print(f"   Keys: {list(annotations.keys())}")
        
        if 'images' in annotations:
            print(f"   Images: {len(annotations['images'])} entries")
            if annotations['images']:
                sample_img = annotations['images'][0]
                print(f"   Sample image entry: {sample_img}")
                
                # Check image dimensions
                widths = [img['width'] for img in annotations['images']]
                heights = [img['height'] for img in annotations['images']]
                print(f"   Image dimensions - Width: {min(widths)}-{max(widths)} (avg: {sum(widths)/len(widths):.0f})")
                print(f"                    - Height: {min(heights)}-{max(heights)} (avg: {sum(heights)/len(heights):.0f})")
        
        if 'categories' in annotations:
            print(f"   Categories: {len(annotations['categories'])} entries")
            if annotations['categories']:
                print(f"   Category ID range: {min(cat['id'] for cat in annotations['categories'])}-{max(cat['id'] for cat in annotations['categories'])}")
                print(f"   Sample categories: {[(cat['id'], cat['name']) for cat in annotations['categories'][:5]]}")
        
        if 'annotations' in annotations:
            print(f"   Annotations: {len(annotations['annotations'])} entries")
            if annotations['annotations']:
                sample_ann = annotations['annotations'][0]
                print(f"   Sample annotation: {sample_ann}")
                
                # Annotations per image
                img_ann_count = Counter(ann['image_id'] for ann in annotations['annotations'])
                ann_counts = list(img_ann_count.values())
                print(f"   Annotations per image: {min(ann_counts)}-{max(ann_counts)} (avg: {sum(ann_counts)/len(ann_counts):.1f})")
                
                # Category frequency
                cat_freq = Counter(ann['category_id'] for ann in annotations['annotations'])
                print(f"   Most frequent categories: {cat_freq.most_common(10)}")
                print(f"   Least frequent categories: {cat_freq.most_common()[-10:]}")
                
                # Check for 'corrected' field
                corrected_count = sum(1 for ann in annotations['annotations'] if ann.get('corrected', False))
                print(f"   Corrected annotations: {corrected_count}/{len(annotations['annotations'])} ({100*corrected_count/len(annotations['annotations']):.1f}%)")
    else:
        print(f"\n2. Annotations file not found: {annotations_path}")
    
    # 3. Check for existing splits
    train_split_path = Path("data/train_split.json")
    val_split_path = Path("data/val_split.json")
    
    print(f"\n3. Existing splits:")
    if train_split_path.exists():
        with open(train_split_path, 'r') as f:
            train_split = json.load(f)
        print(f"   Train split: {len(train_split.get('images', []))} images, {len(train_split.get('annotations', []))} annotations")
    else:
        print(f"   Train split not found: {train_split_path}")
    
    if val_split_path.exists():
        with open(val_split_path, 'r') as f:
            val_split = json.load(f)
        print(f"   Val split: {len(val_split.get('images', []))} images, {len(val_split.get('annotations', []))} annotations")
    else:
        print(f"   Val split not found: {val_split_path}")
    
    # 4. Examine metadata.json
    metadata_path = Path("data/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n4. Metadata structure:")
        print(f"   Keys: {list(metadata.keys())}")
        
        if isinstance(metadata, dict):
            sample_keys = list(metadata.keys())[:3]
            for key in sample_keys:
                print(f"   Sample entry '{key}': {metadata[key]}")
            
            # Count products with different image types
            if sample_keys:
                image_types = set()
                for product_data in metadata.values():
                    if isinstance(product_data, dict) and 'available_images' in product_data:
                        image_types.update(product_data['available_images'])
                print(f"   Available image types: {sorted(image_types)}")
    else:
        print(f"\n4. Metadata file not found: {metadata_path}")
    
    # 5. Examine products directory
    products_dir = Path("data/products")
    if products_dir.exists():
        product_dirs = [d for d in products_dir.iterdir() if d.is_dir()]
        print(f"\n5. Products directory: {len(product_dirs)} product folders")
        
        if product_dirs:
            # Sample a few product directories
            sample_products = product_dirs[:3]
            for prod_dir in sample_products:
                image_files = list(prod_dir.glob("*.jpg")) + list(prod_dir.glob("*.jpeg")) + list(prod_dir.glob("*.png"))
                print(f"   {prod_dir.name}: {len(image_files)} images - {[f.name for f in image_files]}")
            
            # Count total reference images
            total_ref_images = 0
            image_type_counts = Counter()
            for prod_dir in product_dirs:
                for img_file in prod_dir.glob("*.jpg"):
                    total_ref_images += 1
                    image_type_counts[img_file.stem] += 1
                for img_file in prod_dir.glob("*.jpeg"):
                    total_ref_images += 1
                    image_type_counts[img_file.stem] += 1
                for img_file in prod_dir.glob("*.png"):
                    total_ref_images += 1
                    image_type_counts[img_file.stem] += 1
            
            print(f"   Total reference images: {total_ref_images}")
            print(f"   Image type distribution: {dict(image_type_counts.most_common())}")
    else:
        print(f"\n5. Products directory not found: {products_dir}")
    
    # 6. Check for store sections in image metadata
    if annotations_path.exists():
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        print(f"\n6. Store sections analysis:")
        if 'images' in annotations and annotations['images']:
            # Look for section information in image metadata
            sections = set()
            for img in annotations['images']:
                if 'section' in img:
                    sections.add(img['section'])
                # Also check filename patterns
                filename = img.get('file_name', '')
                for section in ['Egg', 'Frokost', 'Knekkebrod', 'Varmedrikker']:
                    if section.lower() in filename.lower():
                        sections.add(section)
            
            if sections:
                print(f"   Found sections: {sorted(sections)}")
                
                # Count images per section
                section_counts = Counter()
                for img in annotations['images']:
                    img_section = img.get('section')
                    if not img_section:
                        filename = img.get('file_name', '')
                        for section in ['Egg', 'Frokost', 'Knekkebrod', 'Varmedrikker']:
                            if section.lower() in filename.lower():
                                img_section = section
                                break
                    if img_section:
                        section_counts[img_section] += 1
                
                print(f"   Images per section: {dict(section_counts)}")
            else:
                print(f"   No explicit section information found in image metadata")
                print(f"   Sample filenames: {[img['file_name'] for img in annotations['images'][:5]]}")

def main():
    explore_data()

if __name__ == "__main__":
    main()
