# Research Learnings


## Experiment 1 — Explore dataset structure and annotations
- Status: FAILED
- Hypothesis: Understanding dataset structure would inform model selection and training approach
- Change: Added comprehensive dataset exploration code to analyze images, annotations, metadata, and products
- Result: ModuleNotFoundError for numpy - required dependencies not installed in environment
- Side effects: None (code didn't execute)
- Takeaway: Need to install required Python packages (numpy, PIL, json) before running dataset exploration code.

## Experiment 1 — Explore data structure and annotations
- Status: SUCCESS
- Hypothesis: Understanding the dataset structure and annotation format to inform model development strategy
- Change: Added comprehensive data exploration script analyzing images, COCO annotations, reference products, and metadata
- Result: Dataset contains 248 store images with 22,731 product annotations across 356 categories, plus 1,599 reference product images from multiple viewpoints
- Side effects: Identified significant class imbalance (84 categories have ≤5 annotations, 23.6% of total) and discovered store section classification from filenames is not viable (0 images properly categorized)
- Takeaway: The dataset has severe class imbalance issues that will require careful handling, and the task should focus on product detection rather than store section classification.

## Experiment 2 — Explore dataset and environment without numpy
- Status: SUCCESS
- Hypothesis: We can understand the dataset structure using only stdlib (json, pathlib, statistics) and check what packages are available
- Change: Rewrote scratch.py to use only Python stdlib, removed numpy/PIL dependencies, added comprehensive environment and dataset analysis
- Result: Successfully analyzed COCO dataset (248 images, 22,731 annotations, 356 categories) and confirmed environment readiness
- Side effects: None observed
- Takeaway: Dataset is properly formatted COCO with significant class imbalance (41 categories have ≤1 annotations), ready for YOLOv8 training after package installation.

## Experiment 2 — Setup environment and create train/val split
- Status: SUCCESS
- Hypothesis: We can create a proper 90/10 stratified split and verify all dependencies work
- Change: Implemented environment setup with dependency installation and created 90/10 train/val split (224/24 images) with COCO format validation
- Result: Split created successfully with 0.097 validation ratio, but PIL installation failed (dependencies_ok=0.0 vs target 1.0)
- Side effects: All images were categorized as "unassigned" section, preventing true stratification by store section
- Takeaway: The split is functional for training despite PIL failure (Pillow likely already available), but store section metadata appears missing from the dataset.

## Experiment 3 — Setup pipeline + YOLOv8n baseline at 640px
- Status: FAILED
- Hypothesis: A quick YOLOv8n baseline with 356 classes at 640px establishes metrics flow and a baseline val_score > 0.05
- Change: Implemented complete YOLOv8n training pipeline with COCO annotation loading, 90/10 train/val split, YOLO format conversion, and model training setup
- Result: Training failed with "operator torchvision::nms does not exist" error, val_score=0.0
- Side effects: Pipeline infrastructure successfully created (data loading, splitting, format conversion all working)
- Takeaway: The torchvision installation failed due to dependency conflicts, preventing YOLOv8 training from completing - need to fix PyTorch/torchvision compatibility before proceeding.

## Experiment 3 — Write evaluation function and convert to YOLO format
- Status: FAILED
- Hypothesis: We can implement the val_score metric and prepare YOLO training data
- Change: Implemented pycocotools-based evaluation function and YOLO format conversion with 5-epoch training test
- Result: Training failed with "operator torchvision::nms does not exist" error, val_score=0.0
- Side effects: Package installation issues with torch/torchvision compatibility
- Takeaway: The evaluation function works but YOLOv8 training fails due to torchvision operator compatibility issues that need to be resolved before proceeding.

## Experiment 4 — YOLOv8m baseline at 640px with nc=357
- Status: FAILED
- Hypothesis: YOLOv8m provides a reasonable baseline, achieving val_score > 0.15
- Change: Modified main.py to train YOLOv8m with nc=357, imgsz=640, epochs=100, batch=16, added early stopping and proper CUDA package installation
- Result: Training failed with "operator torchvision::nms does not exist" error, val_score=0.0
- Side effects: Package installation issues with torch/torchvision compatibility
- Takeaway: PyTorch/torchvision version mismatch is preventing YOLO training; need to fix package compatibility before proceeding with model experiments.

## Experiment 4 — YOLOv8m nc=356 at 640px with longer training
- Status: FAILED
- Hypothesis: Medium model with more epochs improves val_score significantly over nano baseline
- Change: Switched to YOLOv8m, increased epochs to 100, batch=16, fixed nc=356, added CUDA index URL for torch/torchvision
- Result: Training failed with "operator torchvision::nms does not exist" error, val_score=0.0
- Side effects: Package installation issues with torch/torchvision uninstall conflicts
- Takeaway: The torchvision NMS operator error suggests a compatibility issue between installed PyTorch/torchvision versions and YOLO requirements that must be resolved before testing larger models.

## Experiment 5 — Diagnose torch/torchvision environment and fix NMS operator
- Status: FAILED
- Hypothesis: The NMS operator error is caused by version mismatch between pre-installed torch and torchvision
- Change: Created diagnosis script to check torch/torchvision versions and reinstall compatible versions (torch==2.6.0, torchvision==0.21.0)
- Result: Installation failed due to pip being unable to uninstall pre-installed torch (no RECORD file), same NMS operator error persists
- Side effects: None - installation never completed
- Takeaway: The pre-installed torch cannot be uninstalled normally; need to use --force-reinstall --no-deps as suggested in the error message.

## Experiment 5 — Diagnose torch/torchvision compatibility and find working setup
- Status: EXPLORE
- Hypothesis: The NMS operator error is caused by mismatched torch/torchvision versions; we need to find what's pre-installed and work with it
- Change: Added comprehensive torch/torchvision compatibility diagnosis to scratch.py
- Result: Found torch 2.6.0+cu124 installed but cannot uninstall due to missing RECORD file; torchvision import still causes NMS operator error
- Side effects: Confirmed the environment has a non-standard torch installation that cannot be easily modified
- Takeaway: The torch installation appears to be system-level or conda-managed and cannot be fixed with pip, requiring a different approach to resolve the NMS operator issue.

## Experiment 6 — Force-reinstall torchvision with --no-deps and verify NMS
- Status: FAILED
- Hypothesis: Using pip install --force-reinstall --no-deps for torchvision will overwrite the broken installation without needing to uninstall first, fixing the NMS operator
- Change: Force reinstalled torchvision with --no-deps flag and reduced training to 5 epochs for smoke test
- Result: NMS operator now works correctly, but training failed due to PyTorch 2.10 weights_only=True security change preventing loading of YOLOv8 pretrained weights
- Side effects: Other package installations (ultralytics, timm) still fail due to torch RECORD file issues
- Takeaway: The NMS fix worked, but we need to either downgrade PyTorch or use weights_only=False to load YOLOv8 pretrained weights.
