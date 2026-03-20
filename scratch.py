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
        return False
    
    # Don't try to import torchvision yet if we know it's broken
    print("Skipping torchvision import to avoid NMS operator error")
    
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
    
    return True

def fix_torch_torchvision():
    """Fix torch/torchvision compatibility by reinstalling in correct order"""
    print("\n=== FIXING TORCH/TORCHVISION COMPATIBILITY ===")
    
    commands = [
        # First, completely uninstall both
        [sys.executable, '-m', 'pip', 'uninstall', 'torch', 'torchvision', 'torchaudio', '-y'],
        # Install torch first with CUDA support
        [sys.executable, '-m', 'pip', 'install', 'torch==2.6.0', '--index-url', 'https://download.pytorch.org/whl/cu124'],
        # Then install torchvision
        [sys.executable, '-m', 'pip', 'install', 'torchvision==0.21.0', '--index-url', 'https://download.pytorch.org/whl/cu124'],
    ]
    
    for i, cmd in enumerate(commands, 1):
        try:
            print(f"\nStep {i}: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"Command failed: {result.stderr}")
                return False
            else:
                print("Command succeeded")
        except Exception as e:
            print(f"Command error: {e}")
            return False
    
    return True

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
    if not check_versions():
        print("\n✗ Basic torch check failed")
        return
    
    # Step 2: Try to fix torch/torchvision compatibility
    print("\nAttempting to fix torch/torchvision compatibility...")
    if not fix_torch_torchvision():
        print("\n✗ Failed to fix torch/torchvision")
        return
    
    # Step 3: Test NMS operator after fix
    nms_works = test_nms_operator()
    
    # Step 4: Test ultralytics import
    ultralytics_works = test_ultralytics_import()
    
    if nms_works and ultralytics_works:
        print("\n✓ Everything appears to be working!")
        # Test minimal YOLO
        test_minimal_yolo()
        print("\n✓ All tests passed! Ready for training.")
    else:
        print("\n✗ Issues still detected after fix attempt")
        
        # Final diagnostic info
        print("\n=== FINAL DIAGNOSTIC INFO ===")
        print("The issue is likely one of:")
        print("1. torch/torchvision version mismatch")
        print("2. CUDA version incompatibility")
        print("3. Multiple torch installations conflicting")
        print("4. Missing CUDA libraries")
        
        print("\nRecommended manual fix:")
        print("pip uninstall torch torchvision torchaudio -y")
        print("pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124")
        print("pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124")

if __name__ == "__main__":
    main()