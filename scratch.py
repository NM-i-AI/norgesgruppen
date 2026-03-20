import sys
import subprocess

def check_versions():
    """Check exact versions of torch, torchvision, and CUDA"""
    print("=== VERSION DIAGNOSIS ===")
    
    # Check Python version
    print(f"Python: {sys.version}")
    
    # Check torch
    try:
        import torch
        print(f"torch: {torch.__version__}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version (torch): {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"torch: NOT INSTALLED ({e})")
    
    # Check torchvision
    try:
        import torchvision
        print(f"torchvision: {torchvision.__version__}")
    except ImportError as e:
        print(f"torchvision: NOT INSTALLED ({e})")
    
    # Check CUDA toolkit version if available
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"CUDA toolkit: {line.strip()}")
                    break
    except Exception:
        print("CUDA toolkit: nvcc not found")

def test_nms_operator():
    """Test if torchvision.ops.nms works directly"""
    print("\n=== NMS OPERATOR TEST ===")
    
    try:
        import torch
        import torchvision.ops
        
        # Create dummy data for NMS test
        boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        
        # Test NMS
        keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
        print(f"✓ torchvision.ops.nms works: kept indices {keep}")
        return True
        
    except Exception as e:
        print(f"✗ torchvision.ops.nms failed: {e}")
        return False

def test_ultralytics_import():
    """Test if ultralytics can be imported and basic functionality"""
    print("\n=== ULTRALYTICS TEST ===")
    
    try:
        from ultralytics import YOLO
        print("✓ ultralytics imported successfully")
        
        # Try to create a model (this will download weights if needed)
        try:
            model = YOLO('yolov8n.pt')
            print("✓ YOLOv8n model created successfully")
            return True
        except Exception as e:
            print(f"✗ YOLOv8n model creation failed: {e}")
            return False
            
    except ImportError as e:
        print(f"✗ ultralytics import failed: {e}")
        return False

def try_fix_torch_versions():
    """Try different approaches to fix torch/torchvision compatibility"""
    print("\n=== ATTEMPTING FIXES ===")
    
    fixes = [
        # Fix 1: Uninstall both and reinstall matching versions
        {
            'name': 'Uninstall and reinstall torch+torchvision',
            'commands': [
                [sys.executable, '-m', 'pip', 'uninstall', 'torch', 'torchvision', '-y'],
                [sys.executable, '-m', 'pip', 'install', 'torch==2.6.0', 'torchvision==0.21.0', '--index-url', 'https://download.pytorch.org/whl/cu124']
            ]
        },
        # Fix 2: Force reinstall with exact versions
        {
            'name': 'Force reinstall with exact versions',
            'commands': [
                [sys.executable, '-m', 'pip', 'install', 'torch==2.6.0', 'torchvision==0.21.0', '--force-reinstall', '--index-url', 'https://download.pytorch.org/whl/cu124']
            ]
        },
        # Fix 3: Try CPU-only versions first
        {
            'name': 'Install CPU versions first',
            'commands': [
                [sys.executable, '-m', 'pip', 'uninstall', 'torch', 'torchvision', '-y'],
                [sys.executable, '-m', 'pip', 'install', 'torch==2.6.0', 'torchvision==0.21.0']
            ]
        }
    ]
    
    for i, fix in enumerate(fixes, 1):
        print(f"\nTrying Fix {i}: {fix['name']}")
        
        success = True
        for cmd in fix['commands']:
            try:
                print(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"Command failed: {result.stderr}")
                    success = False
                    break
                else:
                    print("Command succeeded")
            except Exception as e:
                print(f"Command error: {e}")
                success = False
                break
        
        if success:
            print(f"Fix {i} completed, testing...")
            # Test if NMS works now
            if test_nms_operator():
                print(f"✓ Fix {i} successful!")
                return True
            else:
                print(f"✗ Fix {i} didn't resolve the issue")
        else:
            print(f"✗ Fix {i} failed to execute")
    
    return False

def test_minimal_yolo():
    """Test minimal YOLO inference to confirm everything works"""
    print("\n=== MINIMAL YOLO TEST ===")
    
    try:
        from ultralytics import YOLO
        import torch
        import numpy as np
        from PIL import Image
        
        # Create a dummy image
        dummy_img = Image.new('RGB', (640, 640), color='red')
        
        # Create model
        model = YOLO('yolov8n.pt')
        
        # Run inference
        results = model(dummy_img, verbose=False)
        
        print(f"✓ YOLO inference successful: {len(results)} result(s)")
        print(f"✓ Result type: {type(results[0])}")
        
        return True
        
    except Exception as e:
        print(f"✗ YOLO inference failed: {e}")
        return False

def main():
    print("TORCH/TORCHVISION COMPATIBILITY DIAGNOSIS")
    print("="*50)
    
    # Step 1: Check current versions
    check_versions()
    
    # Step 2: Test NMS operator directly
    nms_works = test_nms_operator()
    
    # Step 3: Test ultralytics import
    ultralytics_works = test_ultralytics_import()
    
    if nms_works and ultralytics_works:
        print("\n✓ Everything appears to be working!")
        # Test minimal YOLO
        test_minimal_yolo()
    else:
        print("\n✗ Issues detected, attempting fixes...")
        if try_fix_torch_versions():
            print("\n✓ Fix successful! Testing ultralytics...")
            if test_ultralytics_import():
                test_minimal_yolo()
        else:
            print("\n✗ All fixes failed")
            
            # Final diagnostic info
            print("\n=== FINAL DIAGNOSTIC INFO ===")
            print("The issue is likely one of:")
            print("1. torch/torchvision version mismatch")
            print("2. CUDA version incompatibility")
            print("3. Multiple torch installations conflicting")
            print("4. Missing CUDA libraries")
            
            print("\nRecommended manual fix:")
            print("pip uninstall torch torchvision -y")
            print("pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124")

if __name__ == "__main__":
    main()