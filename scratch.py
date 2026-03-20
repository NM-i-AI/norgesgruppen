import torch
import torchvision
import subprocess
import sys

def check_package_version(package_name):
    """Check if a package is installed and return its version"""
    try:
        if package_name == 'torch':
            return torch.__version__
        elif package_name == 'torchvision':
            return torchvision.__version__
        else:
            result = subprocess.run([sys.executable, '-c', f'import {package_name}; print({package_name}.__version__)'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return "NOT INSTALLED"
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    print("=== Environment Verification ===")
    
    # Check critical packages
    packages = {
        'ultralytics': '8.1.0',
        'torch': '2.6.0+cu124', 
        'torchvision': '0.21.0+cu124',
        'timm': '0.9.12',
        'pycocotools': None,  # Just check if installed
        'ensemble-boxes': None,
        'numpy': None,
        'scipy': None,
        'scikit-learn': None
    }
    
    print("\n--- Package Versions ---")
    for package, expected in packages.items():
        actual = check_package_version(package)
        status = "✓" if actual != "NOT INSTALLED" and actual != "ERROR" else "✗"
        print(f"{status} {package}: {actual}")
        if expected and actual != expected and "ERROR" not in actual:
            print(f"  WARNING: Expected {expected}, got {actual}")
    
    # Check GPU availability
    print("\n--- GPU Information ---")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB VRAM)")
            
        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            print(f"✓ GPU memory allocation test: SUCCESS")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ GPU memory allocation test: FAILED - {e}")
    else:
        print("✗ CUDA not available")
    
    # Check data directory structure
    print("\n--- Data Directory Check ---")
    import os
    from pathlib import Path
    
    data_paths = [
        'data/train/images/',
        'data/train/annotations.json',
        'data/products/',
        'data/metadata.json'
    ]
    
    for path in data_paths:
        p = Path(path)
        if p.exists():
            if p.is_dir():
                count = len(list(p.iterdir()))
                print(f"✓ {path} exists ({count} items)")
            else:
                size_mb = p.stat().st_size / (1024*1024)
                print(f"✓ {path} exists ({size_mb:.1f} MB)")
        else:
            print(f"✗ {path} missing")
    
    print("\n=== Environment Check Complete ===")

if __name__ == "__main__":
    main()