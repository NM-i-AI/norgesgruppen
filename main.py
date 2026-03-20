import json
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import random
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

class ProductCropDataset(Dataset):
    """Dataset for product crops extracted from training images"""
    
    def __init__(self, crops, labels, transform=None):
        self.crops = crops
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.crops)
    
    def __getitem__(self, idx):
        crop = self.crops[idx]
        label = self.labels[idx]
        
        if self.transform:
            crop = self.transform(crop)
        
        return crop, label

def extract_product_crops():
    """Extract product crops from training images using ground truth bounding boxes"""
    print("Extracting product crops from training images...")
    
    # Load COCO annotations
    with open('data/train/annotations.json', 'r') as f:
        coco_data = json.load(f)
    
    # Create image info mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Create category mapping
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    category_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}
    
    crops = []
    labels = []
    
    print(f"Processing {len(coco_data['annotations'])} annotations...")
    
    for i, ann in enumerate(coco_data['annotations']):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(coco_data['annotations'])} annotations")
        
        # Get image info
        image_id = ann['image_id']
        img_info = image_info[image_id]
        img_path = Path('data/train/images') / img_info['file_name']
        
        if not img_path.exists():
            continue
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
        
        # Extract crop using bbox
        bbox = ann['bbox']  # [x, y, width, height]
        x, y, w, h = bbox
        
        # Add small padding and ensure within image bounds
        padding = 5
        x1 = max(0, int(x - padding))
        y1 = max(0, int(y - padding))
        x2 = min(image.width, int(x + w + padding))
        y2 = min(image.height, int(y + h + padding))
        
        # Skip very small crops
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue
        
        # Extract crop
        crop = image.crop((x1, y1, x2, y2))
        
        # Get label (YOLO class ID)
        coco_cat_id = ann['category_id']
        yolo_class_id = category_mapping[coco_cat_id]
        
        crops.append(crop)
        labels.append(yolo_class_id)
    
    print(f"Extracted {len(crops)} crops from {len(set(ann['image_id'] for ann in coco_data['annotations']))} images")
    print(f"Number of unique classes: {len(set(labels))}")
    
    return crops, labels, len(categories)

def train_crop_classifier(crops, labels, num_classes):
    """Train a lightweight classifier on product crops"""
    print(f"Training crop classifier on {len(crops)} crops with {num_classes} classes...")
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Split data into train/val
    indices = list(range(len(crops)))
    random.seed(42)
    random.shuffle(indices)
    
    split_idx = int(0.9 * len(indices))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create datasets
    train_crops = [crops[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_crops = [crops[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    train_dataset = ProductCropDataset(train_crops, train_labels, train_transform)
    val_dataset = ProductCropDataset(val_crops, val_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Train crops: {len(train_crops)}, Val crops: {len(val_crops)}")
    
    # Create model - EfficientNet-B0 for speed
    model = models.efficientnet_b0(pretrained=True)
    model.classifier = nn.Linear(model.classifier[1].in_features, num_classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # Training loop
    num_epochs = 20
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_crop_classifier.pth')
            print(f"  New best validation accuracy: {best_val_acc:.2f}%")
        
        scheduler.step()
    
    # Load best model
    model.load_state_dict(torch.load('best_crop_classifier.pth'))
    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, val_transform

def convert_coco_to_yolo_multiclass():
    """Convert COCO annotations to YOLO format with all categories - ALL IMAGES FOR TRAINING"""
    print("Converting COCO annotations to YOLO format (multi-class, all images for training)...")
    
    # Load COCO annotations
    with open('data/train/annotations.json', 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    yolo_dir = Path('data/yolo_multiclass_full')
    yolo_dir.mkdir(exist_ok=True)
    (yolo_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (yolo_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (yolo_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (yolo_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    # Create image_id to filename mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Create category mapping (COCO category_id to YOLO class_id)
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    category_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}
    category_names = [cat['name'] for cat in categories]
    
    print(f"Found {len(categories)} categories")
    
    # Define validation set (same as before for consistent evaluation)
    image_ids = list(image_info.keys())
    random.seed(42)  # For reproducibility
    random.shuffle(image_ids)
    
    split_idx = int(0.9 * len(image_ids))
    val_ids = image_ids[split_idx:]  # Keep same val set for evaluation
    
    # Use ALL images for training
    train_ids = image_ids  # All images go to training
    
    print(f"Train images: {len(train_ids)} (ALL), Val images for eval: {len(val_ids)}")
    
    def process_split(image_ids, split_name):
        """Process train or val split"""
        for image_id in image_ids:
            img_info = image_info[image_id]
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Copy image (create symlink to save space)
            src_path = Path('data/train/images') / img_filename
            dst_path = yolo_dir / 'images' / split_name / img_filename
            
            if src_path.exists():
                # Create symlink instead of copying to save space
                if not dst_path.exists():
                    dst_path.symlink_to(src_path.resolve())
            
            # Convert annotations to YOLO format
            yolo_annotations = []
            for ann in annotations_by_image[image_id]:
                # COCO bbox format: [x, y, width, height] (top-left corner)
                x, y, w, h = ann['bbox']
                
                # Convert to YOLO format: [class_id, x_center, y_center, width, height] (normalized)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                
                # Map COCO category_id to YOLO class_id
                coco_cat_id = ann['category_id']
                yolo_class_id = category_mapping[coco_cat_id]
                
                yolo_annotations.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
            
            # Write YOLO label file
            label_filename = img_filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
            label_path = yolo_dir / 'labels' / split_name / label_filename
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
    
    # Process all images as training data
    process_split(train_ids, 'train')
    
    # Create dataset.yaml
    dataset_yaml = {
        'path': str(yolo_dir.resolve()),
        'train': 'images/train',
        'val': 'images/train',  # Point to train since we're not using YOLO's val split
        'nc': len(categories),  # number of classes
        'names': category_names  # class names
    }
    
    with open(yolo_dir / 'dataset.yaml', 'w') as f:
        import yaml
        yaml.dump(dataset_yaml, f)
    
    print(f"Multi-class YOLO dataset created at {yolo_dir}")
    return yolo_dir / 'dataset.yaml', category_mapping, val_ids

def train_yolo_detector(dataset_yaml_path):
    """Train YOLOv8l detector (faster training for two-stage approach)"""
    print("Training YOLOv8l detector for two-stage approach...")
    
    # Initialize YOLOv8l model
    model = YOLO('yolov8l.pt')
    
    # Reduced training for faster iteration in two-stage approach
    results = model.train(
        data=str(dataset_yaml_path),
        epochs=40,  # Reduced from 80 for faster training
        imgsz=1280,
        batch=6,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='runs/detect',
        name='two_stage_detector',
        save=True,
        save_period=20,
        val=False,
        plots=True,
        verbose=True,
        patience=15,  # Reduced patience
        
        # Detection-specific parameters
        max_det=300,
        conf=0.001,
        iou=0.7,
        
        # Learning rate schedule
        lr0=0.01,
        lrf=0.01,
        
        # Optimizer settings
        optimizer='AdamW',
        weight_decay=0.0005,
        
        # Enhanced data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.9,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        
        # Warmup settings
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Loss function weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Close mosaic augmentation in final epochs
        close_mosaic=10
    )
    
    return model, results

def evaluate_two_stage_model(yolo_model, crop_classifier, crop_transform, val_ids, category_mapping):
    """Evaluate two-stage model: YOLO detection + crop classification"""
    print(f"Evaluating two-stage model on {len(val_ids)} validation images...")
    
    # Load validation data
    with open('data/train/annotations.json', 'r') as f:
        coco_data = json.load(f)
    
    # Create image info mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Group ground truth annotations by image
    gt_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        if ann['image_id'] in val_ids:
            gt_by_image[ann['image_id']].append(ann)
    
    # Run two-stage inference
    all_predictions = []
    all_gt_detection = []
    all_gt_classification = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    crop_classifier.eval()
    
    # Evaluate on subset for speed
    eval_ids = val_ids[:30]  # Evaluate on 30 val images
    
    for i, image_id in enumerate(eval_ids):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(eval_ids)}")
        
        img_info = image_info[image_id]
        img_path = Path('data/train/images') / img_info['file_name']
        
        if not img_path.exists():
            continue
        
        # Stage 1: YOLO detection
        yolo_results = yolo_model.predict(
            source=str(img_path),
            imgsz=1280,
            conf=0.25,
            iou=0.7,
            max_det=300,
            verbose=False
        )
        
        # Stage 2: Crop classification
        predictions = []
        
        if yolo_results and len(yolo_results) > 0 and yolo_results[0].boxes is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = yolo_results[0].boxes.conf.cpu().numpy()
            
            # Load image for cropping
            image = Image.open(img_path).convert('RGB')
            
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                
                # Extract crop
                crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
                
                # Classify crop
                crop_tensor = crop_transform(crop).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    crop_output = crop_classifier(crop_tensor)
                    crop_probs = torch.softmax(crop_output, dim=1)
                    crop_conf, crop_class = torch.max(crop_probs, dim=1)
                    
                    # Combine YOLO detection confidence with crop classification confidence
                    final_score = score * crop_conf.item()
                    final_class = crop_class.item()
                
                # Convert to COCO format
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                
                predictions.append({
                    'bbox': [x, y, w, h],
                    'score': final_score,
                    'category_id': final_class
                })
        
        all_predictions.append(predictions)
        
        # Process ground truth (same as before)
        gt_detection = []
        gt_classification = []
        
        for ann in gt_by_image[image_id]:
            bbox = ann['bbox']
            
            # Detection ground truth (class-agnostic)
            gt_detection.append({
                'bbox': bbox,
                'category_id': 0
            })
            
            # Classification ground truth (class-aware)
            yolo_class_id = category_mapping[ann['category_id']]
            gt_classification.append({
                'bbox': bbox,
                'category_id': yolo_class_id
            })
        
        all_gt_detection.append(gt_detection)
        all_gt_classification.append(gt_classification)
    
    # Calculate metrics (same as before)
    def calculate_iou(box1, box2):
        """Calculate IoU between two boxes in [x, y, w, h] format"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to [x1, y1, x2, y2]
        box1_xyxy = [x1, y1, x1 + w1, y1 + h1]
        box2_xyxy = [x2, y2, x2 + w2, y2 + h2]
        
        # Calculate intersection
        x_left = max(box1_xyxy[0], box2_xyxy[0])
        y_top = max(box1_xyxy[1], box2_xyxy[1])
        x_right = min(box1_xyxy[2], box2_xyxy[2])
        y_bottom = min(box1_xyxy[3], box2_xyxy[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_map_simple(predictions_list, gt_list, iou_threshold=0.5):
        """Simplified mAP calculation"""
        total_tp = 0
        total_fp = 0
        total_gt = 0
        
        for preds, gts in zip(predictions_list, gt_list):
            total_gt += len(gts)
            
            # Sort predictions by score
            preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)
            
            matched_gt = set()
            
            for pred in preds_sorted:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gts):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = calculate_iou(pred['bbox'], gt['bbox'])
                    
                    # For classification, also check category match
                    category_match = (pred['category_id'] == gt['category_id'])
                    
                    if iou > best_iou and iou >= iou_threshold and category_match:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx >= 0:
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        
        return precision, recall
    
    # Calculate detection mAP (class-agnostic)
    detection_predictions = []
    for preds in all_predictions:
        det_preds = []
        for pred in preds:
            det_pred = pred.copy()
            det_pred['category_id'] = 0  # All as single class
            det_preds.append(det_pred)
        detection_predictions.append(det_preds)
    
    detection_precision, detection_recall = calculate_map_simple(
        detection_predictions, all_gt_detection, iou_threshold=0.5
    )
    
    # Calculate classification mAP (class-aware)
    classification_precision, classification_recall = calculate_map_simple(
        all_predictions, all_gt_classification, iou_threshold=0.5
    )
    
    detection_map50 = detection_precision
    classification_map50 = classification_precision
    
    return detection_map50, classification_map50, detection_recall, classification_recall

def main():
    """Main experiment function"""
    print("=== Two-Stage: YOLO Detector + Crop Classifier (Step 7) ===")
    
    start_time = time.time()
    
    try:
        # Step 1: Extract product crops from training images
        print("\n=== Step 1: Extract Product Crops ===")
        crops, labels, num_classes = extract_product_crops()
        
        # Step 2: Train crop classifier
        print("\n=== Step 2: Train Crop Classifier ===")
        crop_classifier, crop_transform = train_crop_classifier(crops, labels, num_classes)
        
        # Step 3: Convert COCO to YOLO format
        print("\n=== Step 3: Prepare YOLO Dataset ===")
        dataset_yaml_path, category_mapping, val_ids = convert_coco_to_yolo_multiclass()
        
        # Step 4: Train YOLO detector
        print("\n=== Step 4: Train YOLO Detector ===")
        yolo_model, train_results = train_yolo_detector(dataset_yaml_path)
        
        # Step 5: Evaluate two-stage model
        print("\n=== Step 5: Evaluate Two-Stage Model ===")
        detection_map50, classification_map50, detection_recall, classification_recall = evaluate_two_stage_model(
            yolo_model, crop_classifier, crop_transform, val_ids, category_mapping
        )
        
        # Step 6: Calculate final score
        final_score = 0.7 * detection_map50 + 0.3 * classification_map50
        
        total_time = time.time() - start_time
        
        # Print metrics
        print(f"\n=== Results ===")
        print(f"METRIC:detection_map50={detection_map50:.4f}")
        print(f"METRIC:classification_map50={classification_map50:.4f}")
        print(f"METRIC:final_score={final_score:.4f}")
        print(f"METRIC:detection_recall={detection_recall:.4f}")
        print(f"METRIC:classification_recall={classification_recall:.4f}")
        print(f"METRIC:num_crops_extracted={len(crops)}")
        print(f"METRIC:num_classes={num_classes}")
        print(f"METRIC:total_time_seconds={total_time:.1f}")
        
        # Success criteria check
        target_classification_map50 = 0.75
        target_final_score = 0.80
        
        success = (classification_map50 > target_classification_map50 and 
                  final_score > target_final_score)
        
        if success:
            print(f"\n✅ SUCCESS: classification_map50 ({classification_map50:.4f}) > {target_classification_map50} AND final_score ({final_score:.4f}) > {target_final_score}")
        else:
            print(f"\n❌ BELOW TARGET: classification_map50 ({classification_map50:.4f}) <= {target_classification_map50} OR final_score ({final_score:.4f}) <= {target_final_score}")
        
        # Compare to exp-006 baseline
        exp006_score = 0.8498
        if final_score > exp006_score:
            improvement = ((final_score - exp006_score) / exp006_score) * 100
            print(f"📈 IMPROVEMENT: +{improvement:.1f}% over exp-006 ({exp006_score:.4f})")
        else:
            decline = ((exp006_score - final_score) / exp006_score) * 100
            print(f"📉 DECLINE: -{decline:.1f}% from exp-006 ({exp006_score:.4f})")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Print error metrics
        print(f"METRIC:detection_map50=0.0")
        print(f"METRIC:classification_map50=0.0")
        print(f"METRIC:final_score=0.0")
        print(f"METRIC:error=1")

if __name__ == "__main__":
    main()