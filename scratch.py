import json
import os
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict

def main():
    print("=== Data Structure Exploration ===")
    
    # Check main data directories
    data_path = Path("data")
    if data_path.exists():
        print(f"\nData directory exists: {data_path}")
        for item in data_path.iterdir():
            if item.is_dir():
                count = len(list(item.iterdir())) if item.exists() else 0
                print(f"  {item.name}/: {count} items")
            else:
                size_mb = item.stat().st_size / (1024*1024) if item.exists() else 0
                print(f"  {item.name}: {size_mb:.1f} MB")
    else:
        print("Data directory not found!")
        return
    
    # Check for existing workspace artifacts
    workspace_dirs = ["yolo_mc", "yolo_sc", "splits", "models", "runs"]
    print("\n=== Existing Workspace Artifacts ===")
    for dirname in workspace_dirs:
        dirpath = Path(dirname)
        if dirpath.exists():
            items = list(dirpath.rglob("*"))
            print(f"  {dirname}/: {len(items)} total items")
            # Show structure
            for item in sorted(dirpath.iterdir())[:10]:  # First 10 items
                if item.is_dir():
                    sub_count = len(list(item.iterdir()))
                    print(f"    {item.name}/: {sub_count} items")
                else:
                    size_mb = item.stat().st_size / (1024*1024)
                    print(f"    {item.name}: {size_mb:.1f} MB")
            if len(list(dirpath.iterdir())) > 10:
                print(f"    ... and {len(list(dirpath.iterdir())) - 10} more")
        else:
            print(f"  {dirname}/: not found")
    
    # Examine annotations.json
    annotations_path = data_path / "train" / "annotations.json"
    if annotations_path.exists():
        print("\n=== Annotations Analysis ===")
        with open(annotations_path) as f:
            coco_data = json.load(f)
        
        print(f"Images: {len(coco_data['images'])}")
        print(f"Annotations: {len(coco_data['annotations'])}")
        print(f"Categories: {len(coco_data['categories'])}")
        
        # Image size distribution
        image_sizes = [(img['width'], img['height']) for img in coco_data['images']]
        size_counter = Counter(image_sizes)
        print(f"\nImage sizes:")
        for size, count in size_counter.most_common(5):
            print(f"  {size[0]}x{size[1]}: {count} images")
        
        # Annotations per image
        img_ann_count = Counter(ann['image_id'] for ann in coco_data['annotations'])
        ann_counts = list(img_ann_count.values())
        print(f"\nAnnotations per image:")
        print(f"  Mean: {np.mean(ann_counts):.1f}")
        print(f"  Median: {np.median(ann_counts):.1f}")
        print(f"  Min: {min(ann_counts)}, Max: {max(ann_counts)}")
        
        # Category distribution
        cat_ann_count = Counter(ann['category_id'] for ann in coco_data['annotations'])
        print(f"\nCategory distribution:")
        print(f"  Categories with 1 annotation: {sum(1 for count in cat_ann_count.values() if count == 1)}")
        print(f"  Categories with 2-5 annotations: {sum(1 for count in cat_ann_count.values() if 2 <= count <= 5)}")
        print(f"  Categories with 6-20 annotations: {sum(1 for count in cat_ann_count.values() if 6 <= count <= 20)}")
        print(f"  Categories with 21+ annotations: {sum(1 for count in cat_ann_count.values() if count > 20)}")
        
        # Most common categories
        print(f"\nTop 10 most frequent categories:")
        for cat_id, count in cat_ann_count.most_common(10):
            cat_name = next((cat['name'] for cat in coco_data['categories'] if cat['id'] == cat_id), f"cat_{cat_id}")
            print(f"  {cat_id}: {cat_name} ({count} annotations)")
        
        # Store section analysis if available
        if 'images' in coco_data:
            store_sections = defaultdict(int)
            for img in coco_data['images']:
                if 'store_section' in img:
                    store_sections[img['store_section']] += 1
                elif 'file_name' in img:
                    # Try to infer from filename
                    fname = img['file_name'].lower()
                    if 'egg' in fname:
                        store_sections['Egg'] += 1
                    elif 'frokost' in fname:
                        store_sections['Frokost'] += 1
                    elif 'knekkebrod' in fname:
                        store_sections['Knekkebrod'] += 1
                    elif 'varmedrikker' in fname:
                        store_sections['Varmedrikker'] += 1
                    else:
                        store_sections['unknown'] += 1
            
            if store_sections:
                print(f"\nStore sections:")
                for section, count in store_sections.items():
                    print(f"  {section}: {count} images")
    
    # Check metadata.json
    metadata_path = data_path / "metadata.json"
    if metadata_path.exists():
        print("\n=== Metadata Analysis ===")
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        print(f"Products in metadata: {len(metadata)}")
        
        # Sample a few products
        print("\nSample products:")
        for i, (product_code, info) in enumerate(list(metadata.items())[:5]):
            if isinstance(info, dict):
                print(f"  {product_code}: {info.get('name', 'no name')} - {info.get('annotation_count', 0)} annotations")
            else:
                print(f"  {product_code}: {info} (unexpected format)")
    
    # Check products directory
    products_path = data_path / "products"
    if products_path.exists():
        print("\n=== Products Directory ===")
        product_dirs = [d for d in products_path.iterdir() if d.is_dir()]
        print(f"Product directories: {len(product_dirs)}")
        
        # Sample product structure
        if product_dirs:
            sample_product = product_dirs[0]
            images = list(sample_product.glob("*.jpg"))
            print(f"\nSample product {sample_product.name}:")
            for img in images[:5]:
                size_kb = img.stat().st_size / 1024
                print(f"  {img.name}: {size_kb:.1f} KB")
    
    print("\n=== Exploration Complete ===")

if __name__ == "__main__":
    main()