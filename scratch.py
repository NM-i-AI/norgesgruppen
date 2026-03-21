import json
import os
import subprocess
import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

def main():
    print("=== ENVIRONMENT VERIFICATION ===")
    
    # Check Python version
    print(f"\n1. PYTHON VERSION:")
    print(f"  Python: {sys.version}")
    
    # Check GPU availability
    print(f"\n2. GPU INFORMATION:")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
                # Check current memory usage
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                print(f"    Memory allocated: {allocated:.1f} GB")
                print(f"    Memory cached: {cached:.1f} GB")
        else:
            print("  No CUDA GPUs available")
    except ImportError:
        print("  PyTorch not installed")
    
    # Check critical package versions
    print(f"\n3. CRITICAL PACKAGE VERSIONS:")
    critical_packages = [
        'ultralytics',
        'torch', 
        'torchvision',
        'timm',
        'pycocotools',
        'ensemble-boxes',
        'supervision',
        'albumentations',
        'opencv-python',
        'opencv-python-headless',
        'Pillow',
        'numpy',
        'scipy',
        'scikit-learn',
        'safetensors',
        'onnxruntime-gpu'
    ]
    
    for package in critical_packages:
        try:
            if package == 'opencv-python' or package == 'opencv-python-headless':
                import cv2
                version = cv2.__version__
                print(f"  {package}: {version}")
            elif package == 'Pillow':
                from PIL import Image
                version = Image.__version__ if hasattr(Image, '__version__') else 'unknown'
                print(f"  {package}: {version}")
            elif package == 'pycocotools':
                import pycocotools
                version = getattr(pycocotools, '__version__', 'unknown')
                print(f"  {package}: {version}")
            else:
                module = __import__(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                print(f"  {package}: {version}")
        except ImportError:
            print(f"  {package}: NOT INSTALLED")
        except Exception as e:
            print(f"  {package}: ERROR - {e}")
    
    # Check for version mismatches with sandbox requirements
    print(f"\n4. SANDBOX COMPATIBILITY CHECK:")
    sandbox_versions = {
        'ultralytics': '8.1.0',
        'torch': '2.6.0',
        'torchvision': '0.21.0',
        'timm': '0.9.12',
        'onnxruntime-gpu': '1.20.0'
    }
    
    for package, expected in sandbox_versions.items():
        try:
            if package == 'onnxruntime-gpu':
                import onnxruntime as ort
                actual = ort.__version__
            else:
                module = __import__(package.replace('-', '_'))
                actual = getattr(module, '__version__', 'unknown')
            
            if actual.startswith(expected.split('.')[0]):
                status = "✓ COMPATIBLE"
            else:
                status = "⚠ VERSION MISMATCH"
            print(f"  {package}: {actual} (expected: {expected}) {status}")
        except ImportError:
            print(f"  {package}: NOT INSTALLED (expected: {expected}) ❌")
    
    # Check existing model weights and artifacts
    print(f"\n5. EXISTING MODEL ARTIFACTS:")
    workspace = Path(".")
    
    # Look for weight files
    weight_extensions = ['.pt', '.pth', '.onnx', '.safetensors', '.bin']
    weight_files = []
    for ext in weight_extensions:
        weight_files.extend(list(workspace.glob(f"**/*{ext}")))
    
    if weight_files:
        print(f"  Found {len(weight_files)} weight files:")
        for weight_file in weight_files:
            size_mb = weight_file.stat().st_size / 1024**2
            print(f"    {weight_file}: {size_mb:.1f} MB")
    else:
        print(f"  No existing weight files found")
    
    # Check for YAML config files
    yaml_files = list(workspace.glob("**/*.yaml")) + list(workspace.glob("**/*.yml"))
    if yaml_files:
        print(f"  Found {len(yaml_files)} YAML config files:")
        for yaml_file in yaml_files:
            print(f"    {yaml_file}")
    else:
        print(f"  No YAML config files found")
    
    # Check for existing train/val split files
    print(f"\n6. EXISTING DATA SPLITS:")
    split_patterns = ['*split*.json', '*train*.json', '*val*.json']
    split_files = []
    for pattern in split_patterns:
        split_files.extend(list(workspace.glob(pattern)))
    
    if split_files:
        print(f"  Found {len(split_files)} split files:")
        for split_file in split_files:
            try:
                with open(split_file, 'r') as f:
                    data = json.load(f)
                if 'images' in data:
                    print(f"    {split_file}: {len(data['images'])} images")
                else:
                    print(f"    {split_file}: {len(data)} items")
            except Exception as e:
                print(f"    {split_file}: Error reading - {e}")
    else:
        print(f"  No existing split files found")
    
    # Check runs directory for previous training runs
    print(f"\n7. PREVIOUS TRAINING RUNS:")
    runs_dir = Path("runs")
    if runs_dir.exists():
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if run_dirs:
            print(f"  Found {len(run_dirs)} run directories:")
            for run_dir in sorted(run_dirs)[-10:]:  # Show last 10 runs
                # Check for weights in run directory
                run_weights = list(run_dir.glob("**/*.pt"))
                if run_weights:
                    best_weight = next((w for w in run_weights if 'best' in w.name), run_weights[0])
                    size_mb = best_weight.stat().st_size / 1024**2
                    print(f"    {run_dir.name}: {best_weight.name} ({size_mb:.1f} MB)")
                else:
                    print(f"    {run_dir.name}: No weights found")
        else:
            print(f"  Runs directory exists but is empty")
    else:
        print(f"  No runs directory found")
    
    # Check data directory structure (brief)
    print(f"\n8. DATA DIRECTORY STATUS:")
    data_dir = Path("data")
    if data_dir.exists():
        train_images = data_dir / "train" / "images"
        annotations = data_dir / "train" / "annotations.json"
        products = data_dir / "products"
        metadata = data_dir / "metadata.json"
        
        if train_images.exists():
            image_count = len(list(train_images.glob("*.jpg")))
            print(f"  Train images: {image_count} files")
        
        if annotations.exists():
            with open(annotations, 'r') as f:
                coco_data = json.load(f)
            print(f"  Annotations: {len(coco_data['annotations'])} boxes, {len(coco_data['categories'])} categories")
        
        if products.exists():
            product_dirs = [d for d in products.iterdir() if d.is_dir()]
            print(f"  Product references: {len(product_dirs)} products")
        
        if metadata.exists():
            print(f"  Metadata: Available")
    else:
        print(f"  Data directory not found")
    
    # Memory and disk space check
    print(f"\n9. SYSTEM RESOURCES:")
    try:
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        print(f"  RAM: {memory.total / 1024**3:.1f} GB total, {memory.available / 1024**3:.1f} GB available")
        print(f"  Disk: {disk.total / 1024**3:.1f} GB total, {disk.free / 1024**3:.1f} GB free")
    except ImportError:
        print(f"  psutil not available for system resource check")
    
    # Test basic ultralytics functionality
    print(f"\n10. ULTRALYTICS FUNCTIONALITY TEST:")
    try:
        from ultralytics import YOLO
        print(f"  ✓ ultralytics import successful")
        
        # Try to load a pretrained model (this will download if not cached)
        print(f"  Testing model loading...")
        model = YOLO('yolov8n.pt')
        print(f"  ✓ YOLOv8n model loaded successfully")
        
        # Check model info
        print(f"  Model info: {model.info(verbose=False)}")
        
    except Exception as e:
        print(f"  ❌ ultralytics test failed: {e}")
    
    print(f"\n=== ENVIRONMENT VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    main()