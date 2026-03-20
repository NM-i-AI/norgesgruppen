import json
import os
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image

def main():
    print("=== Dataset Exploration ===")
    
    # 1. Count images in data/train/images/
    images_dir = Path("data/train/images")
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
        print(f"Number of images: {len(image_files)}")
        
        # Sample a few image dimensions
        sample_images = image_files[:5] if len(image_files) >= 5 else image_files
        print("\nSample image dimensions:")
        for img_path in sample_images:
            try:
                with Image.open(img_path) as img:
                    print(f"  {img_path.name}: {img.size[0]}x{img.size[1]}")
            except Exception as e:
                print(f"  {img_path.name}: Error reading - {e}")
    else:
        print("Images directory not found!")
    
    # 2. Parse annotations.json
    annotations_path = Path("data/train/annotations.json")
    if annotations_path.exists():
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        print(f"\n=== Annotations Analysis ===")
        print(f"Number of images in annotations: {len(coco_data['images'])}")
        print(f"Number of annotations: {len(coco_data['annotations'])}")
        print(f"Number of categories: {len(coco_data['categories'])}")
        
        # Annotations per image distribution
        image_id_to_count = Counter()
        category_counts = Counter()
        
        for ann in coco_data['annotations']:
            image_id_to_count[ann['image_id']] += 1
            category_counts[ann['category_id']] += 1
        
        ann_per_image = list(image_id_to_count.values())
        print(f"\nAnnotations per image stats:")
        print(f"  Mean: {np.mean(ann_per_image):.1f}")
        print(f"  Median: {np.median(ann_per_image):.1f}")
        print(f"  Min: {min(ann_per_image)}")
        print(f"  Max: {max(ann_per_image)}")
        print(f"  Total annotations: {sum(ann_per_image)}")
        
        # Category frequency distribution
        print(f"\n=== Category Distribution ===")
        category_freq = list(category_counts.values())
        print(f"Categories with <5 annotations: {sum(1 for count in category_freq if count < 5)}")
        print(f"Categories with <10 annotations: {sum(1 for count in category_freq if count < 10)}")
        print(f"Categories with <20 annotations: {sum(1 for count in category_freq if count < 20)}")
        print(f"Categories with >=50 annotations: {sum(1 for count in category_freq if count >= 50)}")
        
        print(f"\nCategory frequency stats:")
        print(f"  Mean annotations per category: {np.mean(category_freq):.1f}")
        print(f"  Median annotations per category: {np.median(category_freq):.1f}")
        print(f"  Min annotations per category: {min(category_freq)}")
        print(f"  Max annotations per category: {max(category_freq)}")
        
        # Top 10 most frequent categories
        print(f"\nTop 10 most frequent categories:")
        for cat_id, count in category_counts.most_common(10):
            # Find category name
            cat_name = "unknown"
            for cat in coco_data['categories']:
                if cat['id'] == cat_id:
                    cat_name = cat['name']
                    break
            print(f"  Category {cat_id} ({cat_name}): {count} annotations")
        
        # Bottom 10 least frequent categories
        print(f"\nBottom 10 least frequent categories:")
        for cat_id, count in category_counts.most_common()[-10:]:
            cat_name = "unknown"
            for cat in coco_data['categories']:
                if cat['id'] == cat_id:
                    cat_name = cat['name']
                    break
            print(f"  Category {cat_id} ({cat_name}): {count} annotations")
        
        # Check for store sections in image metadata
        print(f"\n=== Image Metadata ===")
        store_sections = Counter()
        for img in coco_data['images']:
            if 'store_section' in img:
                store_sections[img['store_section']] += 1
            elif 'section' in img:
                store_sections[img['section']] += 1
        
        if store_sections:
            print(f"Store sections found:")
            for section, count in store_sections.items():
                print(f"  {section}: {count} images")
        else:
            print("No store section information found in image metadata")
            # Check if filenames contain section info
            section_from_filename = Counter()
            for img in coco_data['images']:
                filename = img['file_name'].lower()
                if 'egg' in filename:
                    section_from_filename['Egg'] += 1
                elif 'frokost' in filename:
                    section_from_filename['Frokost'] += 1
                elif 'knekkebrod' in filename:
                    section_from_filename['Knekkebrod'] += 1
                elif 'varmedrikker' in filename:
                    section_from_filename['Varmedrikker'] += 1
            
            if section_from_filename:
                print(f"Sections inferred from filenames:")
                for section, count in section_from_filename.items():
                    print(f"  {section}: {count} images")
    else:
        print("Annotations file not found!")
    
    # 3. Examine metadata.json
    metadata_path = Path("data/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n=== Metadata.json Structure ===")
        print(f"Top-level keys: {list(metadata.keys())}")
        
        if 'products' in metadata:
            print(f"Number of products in metadata: {len(metadata['products'])}")
            # Sample a few products
            sample_products = list(metadata['products'].items())[:3]
            print(f"\nSample product entries:")
            for product_code, product_info in sample_products:
                print(f"  Product {product_code}:")
                for key, value in product_info.items():
                    if isinstance(value, (list, dict)):
                        print(f"    {key}: {type(value).__name__} with {len(value)} items")
                    else:
                        print(f"    {key}: {value}")
    else:
        print("Metadata file not found!")
    
    # 4. Check data/products/ directory structure
    products_dir = Path("data/products")
    if products_dir.exists():
        print(f"\n=== Products Directory Structure ===")
        product_dirs = [d for d in products_dir.iterdir() if d.is_dir()]
        print(f"Number of product directories: {len(product_dirs)}")
        
        # Count reference images per product
        ref_image_counts = []
        sample_products_dirs = product_dirs[:5] if len(product_dirs) >= 5 else product_dirs
        
        print(f"\nSample product directories:")
        for product_dir in sample_products_dirs:
            image_files = list(product_dir.glob("*.jpg")) + list(product_dir.glob("*.jpeg")) + list(product_dir.glob("*.png"))
            ref_image_counts.append(len(image_files))
            print(f"  {product_dir.name}: {len(image_files)} images")
            if image_files:
                print(f"    Files: {[f.name for f in image_files[:3]]}{'...' if len(image_files) > 3 else ''}")
        
        # Overall stats for all products
        all_ref_counts = []
        for product_dir in product_dirs:
            image_files = list(product_dir.glob("*.jpg")) + list(product_dir.glob("*.jpeg")) + list(product_dir.glob("*.png"))
            all_ref_counts.append(len(image_files))
        
        if all_ref_counts:
            print(f"\nReference images per product stats:")
            print(f"  Mean: {np.mean(all_ref_counts):.1f}")
            print(f"  Median: {np.median(all_ref_counts):.1f}")
            print(f"  Min: {min(all_ref_counts)}")
            print(f"  Max: {max(all_ref_counts)}")
            print(f"  Products with 0 images: {sum(1 for count in all_ref_counts if count == 0)}")
            print(f"  Products with main.jpg: {sum(1 for product_dir in product_dirs if (product_dir / 'main.jpg').exists())}")
    else:
        print("Products directory not found!")
    
    print("\n=== Exploration Complete ===")

if __name__ == "__main__":
    main()