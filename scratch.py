#!/usr/bin/env python3
"""
Package Discovery Script
Probe the environment to discover available ML packages and their versions.
"""

import sys
import importlib
import subprocess

def try_import(package_name, alias=None):
    """Try to import a package and return (success, version, error)"""
    try:
        if alias:
            module = importlib.import_module(package_name)
            globals()[alias] = module
        else:
            module = importlib.import_module(package_name)
        
        # Try to get version
        version = "unknown"
        for attr in ['__version__', 'version', 'VERSION']:
            if hasattr(module, attr):
                version = getattr(module, attr)
                break
        
        return True, version, None
    except ImportError as e:
        return False, None, str(e)
    except Exception as e:
        return False, None, f"Other error: {str(e)}"

def check_pip_available():
    """Check if pip is available and what packages are installed"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)

def main():
    print("=== ENVIRONMENT PACKAGE DISCOVERY ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path[:3]}...")  # First 3 entries
    
    # Core packages mentioned in the problem description
    core_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'), 
        ('numpy', 'np'),
        ('scipy', 'scipy'),
        ('sklearn', 'sklearn'),
        ('cv2', 'cv2'),  # opencv-python-headless
        ('PIL', 'PIL'),  # Pillow
        ('matplotlib', 'plt'),
        ('tqdm', 'tqdm'),
    ]
    
    # ML/Detection specific packages
    ml_packages = [
        ('ultralytics', 'ultralytics'),
        ('pycocotools', 'pycocotools'),
        ('onnxruntime', 'onnxruntime'),
        ('albumentations', 'albumentations'),
        ('supervision', 'supervision'),
        ('timm', 'timm'),
        ('safetensors', 'safetensors'),
        ('ensemble_boxes', 'ensemble_boxes'),
    ]
    
    # Other potentially useful packages
    other_packages = [
        ('json', 'json'),
        ('pathlib', 'pathlib'),
        ('collections', 'collections'),
        ('random', 'random'),
        ('tempfile', 'tempfile'),
        ('shutil', 'shutil'),
        ('os', 'os'),
        ('sys', 'sys'),
        ('subprocess', 'subprocess'),
        ('multiprocessing', 'multiprocessing'),
        ('threading', 'threading'),
        ('pickle', 'pickle'),
        ('yaml', 'yaml'),
        ('requests', 'requests'),
        ('pandas', 'pd'),
        ('seaborn', 'sns'),
        ('plotly', 'plotly'),
        ('mmdet', 'mmdet'),
        ('detectron2', 'detectron2'),
        ('transformers', 'transformers'),
        ('datasets', 'datasets'),
        ('accelerate', 'accelerate'),
        ('wandb', 'wandb'),
        ('tensorboard', 'tensorboard'),
        ('einops', 'einops'),
    ]
    
    print("\n=== CORE PACKAGES ===")
    available_core = []
    for pkg, alias in core_packages:
        success, version, error = try_import(pkg, alias)
        status = "✓" if success else "✗"
        if success:
            print(f"{status} {pkg:20} {version}")
            available_core.append(pkg)
        else:
            print(f"{status} {pkg:20} MISSING ({error})")
    
    print("\n=== ML/DETECTION PACKAGES ===")
    available_ml = []
    for pkg, alias in ml_packages:
        success, version, error = try_import(pkg, alias)
        status = "✓" if success else "✗"
        if success:
            print(f"{status} {pkg:20} {version}")
            available_ml.append(pkg)
        else:
            print(f"{status} {pkg:20} MISSING ({error})")
    
    print("\n=== OTHER PACKAGES ===")
    available_other = []
    for pkg, alias in other_packages:
        success, version, error = try_import(pkg, alias)
        status = "✓" if success else "✗"
        if success:
            print(f"{status} {pkg:20} {version}")
            available_other.append(pkg)
        else:
            print(f"{status} {pkg:20} MISSING ({error})")
    
    # Check specific torch/CUDA info if available
    if 'torch' in available_core:
        print("\n=== TORCH/CUDA INFO ===")
        try:
            import torch
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"Current CUDA device: {torch.cuda.current_device()}")
                print(f"Device name: {torch.cuda.get_device_name()}")
                print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except Exception as e:
            print(f"Error getting torch info: {e}")
    
    # Check torchvision models if available
    if 'torchvision' in available_core:
        print("\n=== TORCHVISION INFO ===")
        try:
            import torchvision
            print(f"Torchvision version: {torchvision.__version__}")
            # Check if models are available
            from torchvision import models
            print("Available pretrained models: resnet50, efficientnet_b0, vit_b_16, etc.")
        except Exception as e:
            print(f"Error getting torchvision info: {e}")
    
    # Check ultralytics models if available
    if 'ultralytics' in available_ml:
        print("\n=== ULTRALYTICS INFO ===")
        try:
            from ultralytics import YOLO
            print("YOLO models available: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x")
            print("RT-DETR models available: rtdetr-l, rtdetr-x")
        except Exception as e:
            print(f"Error getting ultralytics info: {e}")
    
    # Check pip and installed packages
    print("\n=== PIP PACKAGE LIST ===")
    pip_available, pip_output = check_pip_available()
    if pip_available:
        print("pip is available. Installed packages:")
        lines = pip_output.strip().split('\n')
        # Show first 20 packages
        for line in lines[:25]:
            print(f"  {line}")
        if len(lines) > 25:
            print(f"  ... and {len(lines)-25} more packages")
    else:
        print(f"pip not available or error: {pip_output}")
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"Core packages available: {len(available_core)}/{len(core_packages)}")
    print(f"ML packages available: {len(available_ml)}/{len(ml_packages)}")
    print(f"Other packages available: {len(available_other)}/{len(other_packages)}")
    
    print("\nKey findings:")
    if 'torch' in available_core:
        print("✓ PyTorch is available - can use deep learning")
    else:
        print("✗ PyTorch missing - deep learning not possible")
    
    if 'ultralytics' in available_ml:
        print("✓ Ultralytics available - can use YOLO models")
    else:
        print("✗ Ultralytics missing - YOLO not available")
    
    if 'pycocotools' in available_ml:
        print("✓ pycocotools available - can evaluate COCO metrics")
    else:
        print("✗ pycocotools missing - COCO evaluation not possible")
    
    if 'cv2' in available_core:
        print("✓ OpenCV available - can process images")
    else:
        print("✗ OpenCV missing - limited image processing")
    
    # Recommendations
    print("\n=== RECOMMENDATIONS ===")
    missing_critical = []
    if 'torch' not in available_core:
        missing_critical.append('torch')
    if 'ultralytics' not in available_ml:
        missing_critical.append('ultralytics')
    if 'pycocotools' not in available_ml:
        missing_critical.append('pycocotools')
    
    if missing_critical:
        print(f"Critical packages missing: {missing_critical}")
        print("Add to requirements.txt:")
        for pkg in missing_critical:
            if pkg == 'torch':
                print("  torch>=2.0.0")
            elif pkg == 'ultralytics':
                print("  ultralytics==8.1.0")
            elif pkg == 'pycocotools':
                print("  pycocotools>=2.0.0")
    else:
        print("All critical packages are available!")
    
    # Test a simple import to verify
    print("\n=== VERIFICATION TEST ===")
    try:
        import json
        import pathlib
        print("✓ Basic imports work")
        
        # Test file system access
        data_path = pathlib.Path('data')
        if data_path.exists():
            print("✓ Data directory accessible")
        else:
            print("✗ Data directory not found")
        
        # Test numpy if available
        if 'numpy' in available_core:
            import numpy as np
            arr = np.array([1, 2, 3])
            print(f"✓ NumPy working: {arr}")
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")

if __name__ == "__main__":
    main()