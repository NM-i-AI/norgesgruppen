import json
from pathlib import Path
from utils import (
    create_train_val_split, 
    setup_yolo_dataset_structure, 
    test_evaluation_pipeline
)

def main():
    print("=== Building Evaluation Pipeline and Train/Val Split ===")
    
    # 1. Create 90/10 train/val split
    print("\n1. Creating 90/10 stratified train/val split...")
    train_count, val_count = create_train_val_split(
        annotations_path='data/train/annotations.json',
        train_output='train_split.json',
        val_output='val_split.json',
        val_ratio=0.1,
        seed=42
    )
    
    # 2. Setup YOLO dataset structure
    print("\n2. Setting up YOLO dataset structure...")
    setup_yolo_dataset_structure()
    
    # 3. Test evaluation pipeline
    print("\n3. Testing evaluation pipeline...")
    results = test_evaluation_pipeline()
    
    # 4. Verify split quality
    print("\n4. Verifying split quality...")
    
    # Load and analyze splits
    with open('train_split.json', 'r') as f:
        train_data = json.load(f)
    with open('val_split.json', 'r') as f:
        val_data = json.load(f)
    
    # Check store section distribution
    def get_section_distribution(images):
        sections = {'Egg': 0, 'Frokost': 0, 'Knekkebrod': 0, 'Varmedrikker': 0, 'Unknown': 0}
        for img in images:
            filename = img['file_name']
            if 'Egg' in filename:
                sections['Egg'] += 1
            elif 'Frokost' in filename:
                sections['Frokost'] += 1
            elif 'Knekkebrod' in filename:
                sections['Knekkebrod'] += 1
            elif 'Varmedrikker' in filename:
                sections['Varmedrikker'] += 1
            else:
                sections['Unknown'] += 1
        return sections
    
    train_sections = get_section_distribution(train_data['images'])
    val_sections = get_section_distribution(val_data['images'])
    
    print(f"\nStore section distribution:")
    print(f"  Train: {train_sections}")
    print(f"  Val: {val_sections}")
    
    # Check category coverage
    train_cats = set(ann['category_id'] for ann in train_data['annotations'])
    val_cats = set(ann['category_id'] for ann in val_data['annotations'])
    
    print(f"\nCategory coverage:")
    print(f"  Train: {len(train_cats)} categories")
    print(f"  Val: {len(val_cats)} categories")
    print(f"  Overlap: {len(train_cats & val_cats)} categories")
    print(f"  Val-only: {len(val_cats - train_cats)} categories")
    
    # Calculate final metrics
    total_images = train_count + val_count
    val_ratio = val_count / total_images
    
    print(f"\n=== SPLIT SUMMARY ===")
    print(f"Total images: {total_images}")
    print(f"Train images: {train_count} ({(1-val_ratio):.1%})")
    print(f"Val images: {val_count} ({val_ratio:.1%})")
    print(f"Train annotations: {len(train_data['annotations'])}")
    print(f"Val annotations: {len(val_data['annotations'])}")
    
    # Verify YOLO structure exists
    yolo_dirs = [
        'datasets/train/images',
        'datasets/train/labels', 
        'datasets/val/images',
        'datasets/val/labels'
    ]
    
    all_dirs_exist = all(Path(d).exists() for d in yolo_dirs)
    print(f"\nYOLO dataset structure: {'✅' if all_dirs_exist else '❌'}")
    
    if all_dirs_exist:
        for d in yolo_dirs:
            file_count = len(list(Path(d).glob('*')))
            print(f"  {d}: {file_count} files")
    
    # Output metrics
    print(f"\nMETRIC:train_images={train_count}")
    print(f"METRIC:val_images={val_count}")
    print(f"METRIC:val_ratio={val_ratio:.3f}")
    print(f"METRIC:train_annotations={len(train_data['annotations'])}")
    print(f"METRIC:val_annotations={len(val_data['annotations'])}")
    print(f"METRIC:dummy_detection_mAP={results['detection_mAP']:.4f}")
    print(f"METRIC:dummy_classification_mAP={results['classification_mAP']:.4f}")
    print(f"METRIC:dummy_val_score={results['val_score']:.4f}")
    print(f"METRIC:yolo_structure_ready={1 if all_dirs_exist else 0}")
    
    print("\n=== Pipeline Setup Complete ===")
    print("Ready for training experiments!")

if __name__ == "__main__":
    main()