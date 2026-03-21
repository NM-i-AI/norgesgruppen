import os
import json
from pathlib import Path

def main():
    print("=== WORKSPACE EXPLORATION ===")
    
    # Check current directory structure
    print("\n1. Current directory contents:")
    current_dir = Path('.')
    for item in sorted(current_dir.iterdir()):
        if item.is_dir():
            print(f"  📁 {item.name}/")
        else:
            print(f"  📄 {item.name}")
    
    # Check if data directory exists and explore its structure
    data_dir = Path('data')
    if data_dir.exists():
        print("\n2. Data directory structure:")
        for root, dirs, files in os.walk(data_dir):
            level = root.replace(str(data_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}📁 {os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # Limit to first 10 files per directory
                print(f"{subindent}📄 {file}")
            if len(files) > 10:
                print(f"{subindent}... and {len(files) - 10} more files")
    else:
        print("\n2. No data directory found")
    
    # Check for existing annotation files
    print("\n3. Looking for annotation files:")
    annotation_files = [
        'data/train/annotations.json',
        'train_split.json',
        'val_split.json',
        'data/metadata.json'
    ]
    
    for file_path in annotation_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ Found: {file_path}")
            if file_path.endswith('.json'):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        print(f"     Keys: {list(data.keys())}")
                        if 'images' in data:
                            print(f"     Images: {len(data['images'])}")
                        if 'annotations' in data:
                            print(f"     Annotations: {len(data['annotations'])}")
                        if 'categories' in data:
                            print(f"     Categories: {len(data['categories'])}")
                    elif isinstance(data, list):
                        print(f"     List with {len(data)} items")
                except Exception as e:
                    print(f"     Error reading: {e}")
        else:
            print(f"  ❌ Missing: {file_path}")
    
    # Check for YOLO label directories
    print("\n4. Looking for YOLO label directories:")
    possible_label_dirs = [
        'labels_yolov8x_1280',
        'labels_sc',
        'labels',
        'data/train/labels'
    ]
    
    for label_dir in possible_label_dirs:
        path = Path(label_dir)
        if path.exists():
            print(f"  ✅ Found: {label_dir}")
            txt_files = list(path.glob('*.txt'))
            print(f"     Contains {len(txt_files)} .txt files")
        else:
            print(f"  ❌ Missing: {label_dir}")
    
    # Check for model files
    print("\n5. Looking for model files:")
    model_extensions = ['*.pt', '*.onnx', '*.safetensors', '*.bin']
    model_files = []
    
    for ext in model_extensions:
        model_files.extend(list(Path('.').rglob(ext)))
    
    if model_files:
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  ✅ {model_file} ({size_mb:.1f} MB)")
    else:
        print("  ❌ No model files found")
    
    # Check for config files
    print("\n6. Looking for config files:")
    config_extensions = ['*.yaml', '*.yml', '*.cfg']
    config_files = []
    
    for ext in config_extensions:
        config_files.extend(list(Path('.').rglob(ext)))
    
    if config_files:
        for config_file in config_files:
            print(f"  ✅ {config_file}")
            try:
                with open(config_file, 'r') as f:
                    content = f.read()[:200]  # First 200 chars
                print(f"     Preview: {content.replace(chr(10), ' ')[:100]}...")
            except Exception as e:
                print(f"     Error reading: {e}")
    else:
        print("  ❌ No config files found")
    
    # Check for requirements.txt content
    print("\n7. Current requirements.txt:")
    req_file = Path('requirements.txt')
    if req_file.exists():
        with open(req_file, 'r') as f:
            content = f.read().strip()
        if content:
            print(f"  Content: {content}")
        else:
            print("  Empty file")
    
    # Check for any existing Python files
    print("\n8. Python files in workspace:")
    py_files = list(Path('.').glob('*.py'))
    for py_file in py_files:
        print(f"  📄 {py_file.name}")
    
    print("\n=== EXPLORATION COMPLETE ===")

if __name__ == "__main__":
    main()