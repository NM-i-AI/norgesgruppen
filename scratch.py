import subprocess
import sys
from pathlib import Path

def check_package_version(package_name):
    """Check if a package is installed and return its version"""
    try:
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
    print()
    
    # Check critical packages
    packages = {
        'ultralytics': '8.1.0',
        'torch': '2.6.0',
        'torchvision': '0.21.0', 
        'timm': '0.9.12',
        'pycocotools': 'any',
        'ensemble-boxes': 'any'
    }
    
    print("Package Versions:")
    print("-" * 40)
    all_correct = True
    
    for package, expected in packages.items():
        actual = check_package_version(package)
        status = "✓" if (expected == 'any' and actual != "NOT INSTALLED") or actual.startswith(expected) else "✗"
        if status == "✗":
            all_correct = False
        print(f"{package:15} | Expected: {expected:10} | Actual: {actual:15} | {status}")
    
    print()
    
    # Check GPU availability only if torch is available
    print("GPU Information:")
    print("-" * 40)
    torch_available = check_package_version('torch') != "NOT INSTALLED"
    
    if torch_available:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"CUDA Available: ✓")
                print(f"GPU Count: {gpu_count}")
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                    print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                    
                # Test GPU memory allocation
                try:
                    test_tensor = torch.randn(1000, 1000).cuda()
                    print(f"GPU Memory Test: ✓ (allocated test tensor)")
                    del test_tensor
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"GPU Memory Test: ✗ ({str(e)})")
                    all_correct = False
            else:
                print("CUDA Available: ✗")
                all_correct = False
        except Exception as e:
            print(f"GPU Check Failed: ✗ ({str(e)})")
            all_correct = False
    else:
        print("Torch not available - cannot check GPU")
        all_correct = False
    
    print()
    
    # Check data directory structure
    print("Data Directory Check:")
    print("-" * 40)
    
    data_paths = [
        'data/train/images/',
        'data/train/annotations.json',
        'data/products/',
        'data/metadata.json'
    ]
    
    for path in data_paths:
        if Path(path).exists():
            if Path(path).is_dir():
                count = len(list(Path(path).iterdir()))
                print(f"{path:30} | ✓ (directory with {count} items)")
            else:
                size_mb = Path(path).stat().st_size / (1024*1024)
                print(f"{path:30} | ✓ (file, {size_mb:.1f} MB)")
        else:
            print(f"{path:30} | ✗ (not found)")
            all_correct = False
    
    print()
    
    # Summary
    if all_correct:
        print("✓ Environment verification PASSED - ready for experiments")
    else:
        print("✗ Environment verification FAILED - need to fix issues")
        
        # Show installation commands for missing packages
        print("\nTo install missing packages:")
        missing_packages = []
        for package, expected in packages.items():
            actual = check_package_version(package)
            if actual == "NOT INSTALLED" or not actual.startswith(expected.split('.')[0]):
                if package == 'torch':
                    missing_packages.append("pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124")
                elif package == 'torchvision':
                    missing_packages.append("pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124")
                elif package == 'ultralytics':
                    missing_packages.append("pip install ultralytics==8.1.0")
                elif package == 'timm':
                    missing_packages.append("pip install timm==0.9.12")
                elif package == 'pycocotools':
                    missing_packages.append("pip install pycocotools")
                elif package == 'ensemble-boxes':
                    missing_packages.append("pip install ensemble-boxes")
        
        for cmd in missing_packages:
            print(f"  {cmd}")

if __name__ == "__main__":
    main()