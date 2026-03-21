import json
import os
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

def main():
    print("=== WORKSPACE EXPLORATION ===")
    
    # Check workspace structure
    print("\n1. WORKSPACE STRUCTURE:")
    workspace = Path(".")
    for item in sorted(workspace.iterdir()):
        if item.is_dir():
            print(f"  DIR:  {item.name}/")
            # Show contents of key directories
            if item.name in ['data', 'runs', 'models']:
                for subitem in sorted(item.iterdir()):
                    if subitem.is_dir():
                        print(f"    DIR:  {subitem.name}/")
                    else:
                        print(f"    FILE: {subitem.name}")
        else:
            print(f"  FILE: {item.name}")
    
    # Check data directory structure
    data_dir = Path("data")
    if data_dir.exists():
        print("\n2. DATA DIRECTORY STRUCTURE:")
        for root, dirs, files in os.walk(data_dir):
            level = root.replace(str(data_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # Limit to first 10 files per dir
                print(f"{subindent}{file}")
            if len(files) > 10:
                print(f"{subindent}... and {len(files) - 10} more files")
    
    # Check for existing configs
    print("\n3. EXISTING CONFIGS:")
    config_files = list(workspace.glob("*.yaml")) + list(workspace.glob("*.yml"))
    for config in config_files:
        print(f"  Found: {config.name}")
        try:
            with open(config, 'r') as f:
                content = f.read()
                print(f"    Content preview: {content[:200]}...")
        except Exception as e:
            print(f"    Error reading: {e}")
    
    # Examine annotations.json
    annotations_path = data_dir / "train" / "annotations.json"
    if annotations_path.exists():
        print("\n4. ANNOTATIONS ANALYSIS:")
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        print(f"  Images: {len(coco_data['images'])}")
        print(f"  Annotations: {len(coco_data['annotations'])}")
        print(f"  Categories: {len(coco_data['categories'])}")
        
        # Image statistics
        image_sizes = [(img['width'], img['height']) for img in coco_data['images']]
        widths, heights = zip(*image_sizes)
        print(f"  Image sizes: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
        print(f"  Average size: {np.mean(widths):.0f}x{np.mean(heights):.0f}")
        
        # Annotations per image
        img_id_to_anns = defaultdict(int)
        for ann in coco_data['annotations']:
            img_id_to_anns[ann['image_id']] += 1
        
        ann_counts = list(img_id_to_anns.values())
        print(f"  Annotations per image: {min(ann_counts)} to {max(ann_counts)}")
        print(f"  Average annotations per image: {np.mean(ann_counts):.1f}")
        
        # Category distribution
        cat_counts = Counter(ann['category_id'] for ann in coco_data['annotations'])
        print(f"  Most common categories:")
        for cat_id, count in cat_counts.most_common(10):
            cat_name = next((c['name'] for c in coco_data['categories'] if c['id'] == cat_id), 'unknown')
            print(f"    {cat_id}: {cat_name} ({count} annotations)")
        
        print(f"  Least common categories:")
        for cat_id, count in sorted(cat_counts.items(), key=lambda x: x[1])[:10]:
            cat_name = next((c['name'] for c in coco_data['categories'] if c['id'] == cat_id), 'unknown')
            print(f"    {cat_id}: {cat_name} ({count} annotations)")
        
        # Check for store sections in image filenames or metadata
        print(f"\n  Sample image filenames:")
        for img in coco_data['images'][:10]:
            print(f"    {img['file_name']} (id: {img['id']})")
        
        # Look for store section patterns
        sections = defaultdict(list)
        for img in coco_data['images']:
            filename = img['file_name'].lower()
            if 'egg' in filename:
                sections['Egg'].append(img['id'])
            elif 'frokost' in filename:
                sections['Frokost'].append(img['id'])
            elif 'knekkebrod' in filename:
                sections['Knekkebrod'].append(img['id'])
            elif 'varmedrikker' in filename:
                sections['Varmedrikker'].append(img['id'])
            else:
                sections['Other'].append(img['id'])
        
        print(f"\n  Store sections by filename:")
        for section, img_ids in sections.items():
            print(f"    {section}: {len(img_ids)} images")
    
    # Check metadata.json
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        print("\n5. METADATA ANALYSIS:")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"  Keys in metadata: {list(metadata.keys())}")
        if 'products' in metadata:
            print(f"  Number of products: {len(metadata['products'])}")
            # Sample product - fix the bug here
            if isinstance(metadata['products'], dict) and metadata['products']:
                sample_product = next(iter(metadata['products'].values()))
                print(f"  Sample product keys: {list(sample_product.keys())}")
            elif isinstance(metadata['products'], list) and metadata['products']:
                sample_product = metadata['products'][0]
                print(f"  Sample product keys: {list(sample_product.keys())}")
            else:
                print(f"  Products data structure: {type(metadata['products'])}")
    
    # Check products directory
    products_dir = data_dir / "products"
    if products_dir.exists():
        print("\n6. PRODUCTS DIRECTORY:")
        product_dirs = [d for d in products_dir.iterdir() if d.is_dir()]
        print(f"  Number of product directories: {len(product_dirs)}")
        if product_dirs:
            sample_dir = product_dirs[0]
            print(f"  Sample product dir: {sample_dir.name}")
            sample_files = list(sample_dir.iterdir())
            print(f"  Sample files: {[f.name for f in sample_files]}")
    
    # Check for existing train/val splits
    print("\n7. EXISTING SPLITS:")
    split_files = list(workspace.glob("*split*.json")) + list(workspace.glob("*train*.json")) + list(workspace.glob("*val*.json"))
    for split_file in split_files:
        print(f"  Found: {split_file.name}")
    
    # Check runs directory for previous experiments
    runs_dir = Path("runs")
    if runs_dir.exists():
        print("\n8. PREVIOUS RUNS:")
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        print(f"  Number of run directories: {len(run_dirs)}")
        for run_dir in sorted(run_dirs)[-5:]:  # Show last 5 runs
            print(f"    {run_dir.name}")
    
    print("\n=== EXPLORATION COMPLETE ===")

if __name__ == "__main__":
    main()