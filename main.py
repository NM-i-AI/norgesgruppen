#!/usr/bin/env python3
"""
Main experiment file. This is the entry point executed by the orchestrator.
Edit this file to implement experiments.

Print metrics to stdout as: METRIC:name=value
Example: METRIC:val_accuracy=0.74
"""

import subprocess
import sys
import os

def install_package(package_spec):
    """Install a package using pip"""
    try:
        print(f"Installing {package_spec}...")
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', package_spec], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✓ Successfully installed {package_spec}")
            return True
        else:
            print(f"✗ Failed to install {package_spec}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Exception installing {package_spec}: {e}")
        return False

def check_gpu_info():
    """Check GPU availability and info"""
    try:
        import torch
        print(f"\n=== GPU Information ===")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_memory = gpu_props.total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                print(f"  Compute capability: {gpu_props.major}.{gpu_props.minor}")
            
            # Test GPU functionality
            try:
                x = torch.randn(100, 100).cuda()
                y = torch.mm(x, x.t())
                print(f"✓ GPU computation test passed")
                return True
            except Exception as e:
                print(f"✗ GPU computation test failed: {e}")
                return False
        else:
            print("✗ No CUDA GPU available")
            return False
            
    except ImportError:
        print("✗ PyTorch not available")
        return False

def verify_packages():
    """Verify all required packages are installed with correct versions"""
    packages_to_check = {
        'ultralytics': '8.1.0',
        'torch': '2.6.0',
        'torchvision': '0.21.0', 
        'timm': '0.9.12'
    }
    
    print(f"\n=== Package Verification ===")
    all_ok = True
    
    for package, expected_version in packages_to_check.items():
        try:
            if package == 'torch':
                import torch
                version = torch.__version__
            elif package == 'torchvision':
                import torchvision
                version = torchvision.__version__
            elif package == 'ultralytics':
                import ultralytics
                version = ultralytics.__version__
            elif package == 'timm':
                import timm
                version = timm.__version__
            
            # Handle versions with suffixes like +cu124
            base_version = version.split('+')[0]
            if base_version == expected_version or version.startswith(expected_version):
                print(f"✓ {package}: {version} (expected: {expected_version})")
            else:
                print(f"✗ {package}: {version} (expected: {expected_version})")
                all_ok = False
                
        except ImportError:
            print(f"✗ {package}: NOT INSTALLED")
            all_ok = False
    
    return all_ok

def main():
    print("=== Installing Required Dependencies ===")
    
    # List of packages to install in order
    packages = [
        'ultralytics==8.1.0',
        'torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124',
        'torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124', 
        'timm==0.9.12',
        'pycocotools',
        'ensemble-boxes'
    ]
    
    install_success = True
    
    for package in packages:
        if not install_package(package):
            install_success = False
    
    print(f"\n=== Installation Summary ===")
    if install_success:
        print("✓ All packages installed successfully")
    else:
        print("✗ Some packages failed to install")
    
    # Verify installations
    packages_ok = verify_packages()
    
    # Check GPU
    gpu_ok = check_gpu_info()
    
    # Final metrics
    print(f"\n=== Final Status ===")
    if install_success and packages_ok:
        print("METRIC:packages_installed=1.0")
    else:
        print("METRIC:packages_installed=0.0")
    
    if gpu_ok:
        print("METRIC:gpu_available=1.0")
    else:
        print("METRIC:gpu_available=0.0")
    
    # Overall success metric
    if install_success and packages_ok and gpu_ok:
        print("METRIC:setup_success=1.0")
        print("✓ Environment setup complete - ready for training!")
    else:
        print("METRIC:setup_success=0.0")
        print("✗ Environment setup incomplete")

if __name__ == "__main__":
    main()
