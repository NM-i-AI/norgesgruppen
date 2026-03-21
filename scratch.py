import os
import json
from pathlib import Path
import torch
import subprocess
import sys

def main():
    print("=== COMPREHENSIVE DATA & ENVIRONMENT EXPLORATION ===")
    
    # 1. Load and analyze annotations.json
    print("\n1. ANNOTATIONS ANALYSIS:")
    annotations_path = Path('data/train/annotations.json')
    if annotations_path.exists():
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        print(f"  Images: {len(coco_data['images'])}")
        print(f"  Annotations: {len(coco_data['annotations'])}")
        print(f"  Categories: {len(coco_data['categories'])}")
        
        # Per-category annotation counts
        category_counts = {}
        for ann in coco_data['annotations']:
            cat_id = ann['category_id']
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        
        print(f"  Categories with annotations: {len(category_counts)}")
        print(f"  Avg annotations per image: {len(coco_data['annotations']) / len(coco_data['images']):.1f}")
        
        # Category distribution
        sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"  Most frequent categories:")
        for cat_id, count in sorted_cats[:10]:
            cat_name = next((c['name'] for c in coco_data['categories'] if c['id'] == cat_id), f"cat_{cat_id}")
            print(f"    {cat_id}: {cat_name} ({count} annotations)")
        
        print(f"  Least frequent categories:")
        for cat_id, count in sorted_cats[-10:]:
            cat_name = next((c['name'] for c in coco_data['categories'] if c['id'] == cat_id), f"cat_{cat_id}")
            print(f"    {cat_id}: {cat_name} ({count} annotations)")
        
        # Image dimensions analysis
        widths = [img['width'] for img in coco_data['images']]
        heights = [img['height'] for img in coco_data['images']]
        print(f"  Image dimensions:")
        print(f"    Width range: {min(widths)} - {max(widths)} (avg: {sum(widths)/len(widths):.0f})")
        print(f"    Height range: {min(heights)} - {max(heights)} (avg: {sum(heights)/len(heights):.0f})")
        
        # Store sections analysis
        store_sections = {}
        for img in coco_data['images']:
            filename = img['file_name']
            # Extract store section from filename pattern
            if 'Egg' in filename:
                section = 'Egg'
            elif 'Frokost' in filename:
                section = 'Frokost'
            elif 'Knekkebrod' in filename:
                section = 'Knekkebrod'
            elif 'Varmedrikker' in filename:
                section = 'Varmedrikker'
            else:
                section = 'Unknown'
            store_sections[section] = store_sections.get(section, 0) + 1
        
        print(f"  Store sections:")
        for section, count in store_sections.items():
            print(f"    {section}: {count} images")
    
    # 2. Check train/val splits
    print("\n2. TRAIN/VAL SPLITS:")
    train_split_path = Path('train_split.json')
    val_split_path = Path('val_split.json')
    
    if train_split_path.exists():
        with open(train_split_path, 'r') as f:
            train_data = json.load(f)
        print(f"  Train split: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
    else:
        print(f"  ❌ train_split.json not found")
    
    if val_split_path.exists():
        with open(val_split_path, 'r') as f:
            val_data = json.load(f)
        print(f"  Val split: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")
        
        if train_split_path.exists():
            total_images = len(train_data['images']) + len(val_data['images'])
            val_ratio = len(val_data['images']) / total_images
            print(f"  Split ratio: {len(train_data['images'])}/{len(val_data['images'])} ({val_ratio:.1%} validation)")
    else:
        print(f"  ❌ val_split.json not found")
    
    # 3. Explore products directory
    print("\n3. PRODUCTS DIRECTORY:")
    products_dir = Path('data/products')
    if products_dir.exists():
        product_dirs = [d for d in products_dir.iterdir() if d.is_dir()]
        print(f"  Product directories: {len(product_dirs)}")
        
        # Sample a few products to see structure
        image_types = {}
        total_images = 0
        for i, product_dir in enumerate(product_dirs[:10]):
            images = list(product_dir.glob('*.jpg')) + list(product_dir.glob('*.png'))
            total_images += len(images)
            print(f"    {product_dir.name}: {len(images)} images")
            for img in images:
                img_type = img.stem  # e.g., 'main', 'front', 'back'
                image_types[img_type] = image_types.get(img_type, 0) + 1
        
        if len(product_dirs) > 10:
            # Count remaining products
            for product_dir in product_dirs[10:]:
                images = list(product_dir.glob('*.jpg')) + list(product_dir.glob('*.png'))
                total_images += len(images)
                for img in images:
                    img_type = img.stem
                    image_types[img_type] = image_types.get(img_type, 0) + 1
        
        print(f"  Total reference images: {total_images}")
        print(f"  Image types: {dict(sorted(image_types.items(), key=lambda x: x[1], reverse=True))}")
    else:
        print(f"  ❌ data/products directory not found")
    
    # 4. Search for ALL .pt weight files recursively
    print("\n4. TRAINED MODELS SEARCH:")
    model_files = []
    for ext in ['*.pt', '*.onnx', '*.safetensors', '*.bin']:
        model_files.extend(list(Path('.').rglob(ext)))
    
    if model_files:
        print(f"  Found {len(model_files)} model files:")
        for model_file in sorted(model_files):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"    {model_file} ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ No model files found")
    
    # 5. GPU and environment check
    print("\n5. ENVIRONMENT CHECK:")
    
    # GPU availability
    if torch.cuda.is_available():
        print(f"  CUDA available: Yes")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print(f"  CUDA available: No")
    
    # Package versions
    print(f"  Python version: {sys.version.split()[0]}")
    
    packages_to_check = [
        'torch', 'torchvision', 'ultralytics', 'timm', 
        'pycocotools', 'numpy', 'opencv-python', 'albumentations',
        'ensemble-boxes', 'supervision', 'safetensors'
    ]
    
    for package in packages_to_check:
        try:
            result = subprocess.run([sys.executable, '-c', f'import {package}; print({package}.__version__)'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"  {package}: {version}")
            else:
                print(f"  {package}: ❌ Import failed")
        except Exception as e:
            print(f"  {package}: ❌ Error checking version")
    
    # 6. Read ALL YAML config files
    print("\n6. YAML CONFIGURATIONS:")
    yaml_files = list(Path('.').rglob('*.yaml')) + list(Path('.').rglob('*.yml'))
    
    for yaml_file in yaml_files:
        print(f"  📄 {yaml_file}")
        try:
            with open(yaml_file, 'r') as f:
                content = f.read()
            print(f"     Content ({len(content)} chars):")
            # Print first few lines
            lines = content.split('\n')[:10]
            for line in lines:
                if line.strip():
                    print(f"       {line}")
            if len(content.split('\n')) > 10:
                print(f"       ... ({len(content.split('\n')) - 10} more lines)")
        except Exception as e:
            print(f"     Error reading: {e}")
        print()
    
    # 7. Check metadata.json
    print("\n7. METADATA.JSON:")
    metadata_path = Path('data/metadata.json')
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"  Keys: {list(metadata.keys())}")
        
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if isinstance(value, (list, dict)):
                    print(f"    {key}: {type(value).__name__} with {len(value)} items")
                    if isinstance(value, list) and len(value) > 0:
                        print(f"      Sample: {value[0]}")
                    elif isinstance(value, dict) and len(value) > 0:
                        sample_key = list(value.keys())[0]
                        print(f"      Sample: {sample_key}: {value[sample_key]}")
                else:
                    print(f"    {key}: {value}")
    else:
        print(f"  ❌ data/metadata.json not found")
    
    # 8. Check for any existing results or logs
    print("\n8. EXISTING RESULTS/LOGS:")
    result_files = []
    for pattern in ['*.log', '*.txt', 'results*', 'runs/', 'wandb/', 'tensorboard*']:
        result_files.extend(list(Path('.').rglob(pattern)))
    
    if result_files:
        print(f"  Found {len(result_files)} result/log files:")
        for result_file in sorted(result_files)[:20]:  # Limit output
            if result_file.is_file():
                size_kb = result_file.stat().st_size / 1024
                print(f"    📄 {result_file} ({size_kb:.1f} KB)")
            else:
                print(f"    📁 {result_file}/")
        if len(result_files) > 20:
            print(f"    ... and {len(result_files) - 20} more")
    else:
        print(f"  No result/log files found")
    
    print("\n=== COMPREHENSIVE EXPLORATION COMPLETE ===")
    print("\nSUMMARY:")
    print(f"- Dataset: {len(coco_data['images']) if 'coco_data' in locals() else 'Unknown'} images, {len(coco_data['annotations']) if 'coco_data' in locals() else 'Unknown'} annotations")
    print(f"- Categories: {len(coco_data['categories']) if 'coco_data' in locals() else 'Unknown'}")
    print(f"- Products: {len(product_dirs) if 'product_dirs' in locals() else 'Unknown'} with reference images")
    print(f"- Pre-trained models: {len(model_files) if 'model_files' in locals() else 0}")
    print(f"- GPU: {torch.cuda.device_count() if torch.cuda.is_available() else 0} available")
    print(f"- Train/Val splits: {'✅' if train_split_path.exists() and val_split_path.exists() else '❌'}")

if __name__ == "__main__":
    main()