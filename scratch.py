import subprocess
import sys
import json
from pathlib import Path

def check_package_version(package_name):
    """Check if a package is installed and return its version"""
    try:
        result = subprocess.run([sys.executable, '-c', f'import {package_name}; print({package_name}.__version__)'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return None
    except Exception as e:
        return None

def check_torch_info():
    """Check torch installation and CUDA availability"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        return True
    except ImportError:
        print("PyTorch not installed")
        return False

def check_torchvision_info():
    """Check torchvision installation"""
    try:
        import torchvision
        print(f"Torchvision version: {torchvision.__version__}")
        return True
    except ImportError:
        print("Torchvision not installed")
        return False

def main():
    print("=== Environment Check ===\n")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print()
    
    # Check critical packages with required versions
    critical_packages = {
        'ultralytics': '8.1.0',
        'torch': '2.6.0',  # May have +cu124 suffix
        'torchvision': '0.21.0',  # May have +cu124 suffix
        'timm': '0.9.12'
    }
    
    print("=== Critical Package Versions ===")
    all_critical_ok = True
    
    for package, required_version in critical_packages.items():
        installed_version = check_package_version(package)
        if installed_version:
            # Handle torch/torchvision versions that may have +cu124 suffix
            base_version = installed_version.split('+')[0]
            if base_version == required_version or installed_version.startswith(required_version):
                status = "✓ OK"
            else:
                status = "✗ VERSION MISMATCH"
                all_critical_ok = False
            print(f"{package}: {installed_version} (required: {required_version}) {status}")
        else:
            print(f"{package}: NOT INSTALLED ✗")
            all_critical_ok = False
    
    print()
    
    # Check other important packages
    other_packages = [
        'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'tqdm', 'pillow',
        'pycocotools', 'ensemble-boxes', 'supervision', 'albumentations',
        'opencv-python', 'safetensors', 'onnxruntime-gpu'
    ]
    
    print("=== Other Important Packages ===")
    for package in other_packages:
        # Handle package name variations
        import_name = package
        if package == 'pillow':
            import_name = 'PIL'
        elif package == 'opencv-python':
            import_name = 'cv2'
        elif package == 'scikit-learn':
            import_name = 'sklearn'
        elif package == 'onnxruntime-gpu':
            import_name = 'onnxruntime'
        
        version = check_package_version(import_name)
        if version:
            print(f"{package}: {version} ✓")
        else:
            print(f"{package}: NOT INSTALLED")
    
    print()
    
    # Check PyTorch and CUDA
    print("=== PyTorch & CUDA Info ===")
    torch_ok = check_torch_info()
    print()
    
    # Check torchvision
    print("=== Torchvision Info ===")
    torchvision_ok = check_torchvision_info()
    print()
    
    # Check data directory structure
    print("=== Data Directory Check ===")
    data_paths = [
        "data/train/images",
        "data/train/annotations.json",
        "data/products",
        "data/metadata.json"
    ]
    
    data_ok = True
    for path_str in data_paths:
        path = Path(path_str)
        if path.exists():
            if path.is_dir():
                count = len(list(path.iterdir()))
                print(f"{path_str}: EXISTS (directory with {count} items) ✓")
            else:
                print(f"{path_str}: EXISTS (file) ✓")
        else:
            print(f"{path_str}: NOT FOUND ✗")
            data_ok = False
    
    print()
    
    # Summary
    print("=== Summary ===")
    if all_critical_ok:
        print("✓ All critical packages have correct versions")
    else:
        print("✗ Some critical packages need to be installed/updated")
        print("\nTo install correct versions:")
        print("pip install ultralytics==8.1.0")
        print("pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124")
        print("pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124")
        print("pip install timm==0.9.12")
    
    if torch_ok and torch.cuda.is_available():
        print("✓ GPU available for training")
    else:
        print("✗ No GPU available - training will be very slow")
    
    if data_ok:
        print("✓ All required data directories/files found")
    else:
        print("✗ Some data files missing")
    
    print("\n=== Environment Check Complete ===")

if __name__ == "__main__":
    main()