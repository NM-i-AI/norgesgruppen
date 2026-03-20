# Research Learnings


## Experiment 1 — Explore data structure and annotations
- Status: SUCCESS
- Hypothesis: Understanding the dataset structure and characteristics
- Change: Created comprehensive data exploration script analyzing images, annotations, metadata, and product reference images
- Result: Dataset contains 248 shelf images with 22,731 product annotations across 356 categories, plus 1,599 reference product images
- Side effects: None (exploration only)
- Takeaway: This is a dense retail object detection dataset with high annotation density (avg 91.7 products per image) and significant class imbalance that will require careful handling.

## Experiment 2 — Validate splits and check environment
- Status: FAILED
- Hypothesis: Need to confirm splits are valid and GPU/packages are available
- Change: Added validation script to check train/val splits and environment setup
- Result: Train split file missing at data/train_split.json, PyTorch not available
- Side effects: None (script failed before any heavy operations)
- Takeaway: Must create the train/val splits and install PyTorch before proceeding with model training experiments.

## Experiment 3 — Create train/val splits and verify environment
- Status: FAILED
- Hypothesis: We need valid train/val splits (90/10 stratified by image) and a working PyTorch+ultralytics environment before any training
- Change: Implemented train/val split creation with 90/10 stratification by image using seed=42, plus environment validation for PyTorch, CUDA, ultralytics, and other required packages
- Result: Splits created successfully (224 train/24 val images, 9.7% val ratio) but environment validation failed - all required packages (PyTorch, ultralytics, pycocotools, numpy, scipy, scikit-learn, timm) are missing
- Side effects: Category distribution shows 130 categories only in train and 6 only in val, indicating potential class imbalance issues for validation
- Takeaway: The data splits are ready but the Python environment needs all ML packages installed before any training can proceed.

## Experiment 4 — Build evaluation function and YOLO dataset conversion
- Status: FAILED
- Hypothesis: We need a reusable eval function and YOLO-format data before any training
- Change: Implemented evaluation function using pycocotools and COCO-to-YOLO conversion function
- Result: YOLO conversion succeeded (248 images converted), but evaluation function failed due to missing pycocotools dependency
- Side effects: None observed
- Takeaway: Need to install pycocotools dependency before the evaluation function can work, but YOLO dataset conversion is ready for training.

## Experiment 5 — YOLOv8m baseline with nc=356 at imgsz=1280
- Status: FAILED
- Hypothesis: YOLOv8m at 1280 is a good balance of speed and accuracy for 356-class detection on dense shelf images
- Change: Implemented YOLOv8m baseline training pipeline with nc=356 classes at imgsz=1280
- Result: Training failed due to missing ultralytics package (training_success=0.0)
- Side effects: None - experiment did not execute
- Takeaway: The ultralytics package needs to be installed before any YOLO experiments can proceed.

## Experiment 6 — YOLOv8x with nc=356 at imgsz=1280
- Status: FAILED
- Hypothesis: Larger model improves both detection and classification on this dense dataset
- Change: Updated to YOLOv8x model with increased patience (30 epochs) and maintained imgsz=1280
- Result: Training failed due to missing ultralytics package dependency
- Side effects: None (training never started)
- Takeaway: The ultralytics package needs to be installed before any YOLO training can proceed.

## Experiment 7 — YOLOv8x with nc=1 (detection only) at imgsz=1280
- Status: FAILED
- Hypothesis: Single-class detector achieves higher recall since it only needs to find products, not classify them
- Change: Modified main.py to train YOLOv8x with nc=1 and convert annotations to single-class format
- Result: Training failed due to missing ultralytics package dependency
- Side effects: None (experiment didn't run)
- Takeaway: The ultralytics package needs to be installed before any YOLO experiments can proceed.
