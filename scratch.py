import subprocess
import sys

def check_package_version(package_name):
    """Check if a package is installed and return its version."""
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
    
    # Check critical package versions
    packages_to_check = {
        'ultralytics': '8.1.0',  # Required version
        'torch': '2.6.0+cu124',  # Expected version
        'torchvision': '0.21.0+cu124',  # Expected version
        'timm': '0.9.12',  # Expected version
        'pycocotools': None,  # Just check if installed
        'ensemble-boxes': None,  # Just check if installed
        'numpy': None,
        'scipy': None,
        'scikit-learn': None,
        'albumentations': None,
        'supervision': None
    }
    
    print("\n--- Package Versions ---")
    version_issues = []
    missing_packages = []
    
    for package, expected_version in packages_to_check.items():
        actual_version = check_package_version(package)
        status = "✓" if actual_version != "NOT INSTALLED" and "ERROR" not in actual_version else "✗"
        
        print(f"{status} {package}: {actual_version}")
        
        if actual_version == "NOT INSTALLED" or "ERROR" in actual_version:
            missing_packages.append(package)
        elif expected_version and actual_version != expected_version:
            version_issues.append(f"{package}: expected {expected_version}, got {actual_version}")
    
    # Check GPU availability only if torch is available
    print("\n--- GPU Information ---")
    torch_version = check_package_version('torch')
    if torch_version != "NOT INSTALLED" and "ERROR" not in torch_version:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"✓ CUDA available: {torch.version.cuda}")
                print(f"✓ GPU count: {gpu_count}")
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                    print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                    
                # Test GPU memory allocation
                try:
                    test_tensor = torch.randn(1000, 1000).cuda()
                    allocated_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                    print(f"✓ GPU memory test passed. Allocated: {allocated_memory:.3f} GB")
                    del test_tensor
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"✗ GPU memory test failed: {e}")
            else:
                print("✗ CUDA not available")
        except Exception as e:
            print(f"✗ Error checking GPU: {e}")
    else:
        print("✗ Cannot check GPU - torch not installed")
    
    # Check if ultralytics needs to be pinned
    ultralytics_version = check_package_version('ultralytics')
    print("\n--- Critical Version Check ---")
    
    if ultralytics_version == "8.1.0":
        print("✓ ultralytics==8.1.0 correctly installed")
    elif "ERROR" in ultralytics_version or ultralytics_version == "NOT INSTALLED":
        print("✗ ultralytics not installed - need to install ultralytics==8.1.0")
        print("Run: pip install ultralytics==8.1.0")
    else:
        print(f"⚠ ultralytics version mismatch: {ultralytics_version} (need 8.1.0)")
        print("Run: pip install ultralytics==8.1.0")
    
    # Installation instructions
    if missing_packages:
        print("\n--- Installation Instructions ---")
        print("Missing packages detected. Install with:")
        
        # Core packages with specific versions
        if 'torch' in missing_packages or 'torchvision' in missing_packages:
            print("pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124")
        
        if 'ultralytics' in missing_packages:
            print("pip install ultralytics==8.1.0")
            
        if 'timm' in missing_packages:
            print("pip install timm==0.9.12")
            
        # Other packages
        other_missing = [pkg for pkg in missing_packages if pkg not in ['torch', 'torchvision', 'ultralytics', 'timm']]
        if other_missing:
            print(f"pip install {' '.join(other_missing)}")
    
    # Summary
    print("\n--- Summary ---")
    if missing_packages:
        print(f"✗ Missing packages: {', '.join(missing_packages)}")
    elif version_issues:
        print("⚠ Version issues found:")
        for issue in version_issues:
            print(f"  - {issue}")
    else:
        print("✓ All package versions look good")
    
    # Check data directory structure
    print("\n--- Data Directory Check ---")
    import pathlib
    
    data_paths = [
        'data/train/images/',
        'data/train/annotations.json',
        'data/products/',
        'data/metadata.json'
    ]
    
    for path_str in data_paths:
        path = pathlib.Path(path_str)
        if path.exists():
            if path.is_dir():
                try:
                    count = len(list(path.iterdir()))
                    print(f"✓ {path_str} exists ({count} items)")
                except:
                    print(f"✓ {path_str} exists (directory)")
            else:
                print(f"✓ {path_str} exists (file)")
        else:
            print(f"✗ {path_str} not found")

if __name__ == "__main__":
    main()