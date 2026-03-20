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
