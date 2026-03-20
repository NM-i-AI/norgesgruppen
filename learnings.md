# Research Learnings


## Experiment 1 — Explore dataset structure and annotations
- Status: SUCCESS
- Hypothesis: None (exploration)
- Change: Implemented comprehensive dataset exploration script to understand COCO annotations, metadata, and reference images
- Result: Dataset contains 248 images with 22,731 annotations across 356 product categories; severe class imbalance with 84 categories having ≤5 annotations
- Side effects: None
- Takeaway: The dataset has extreme class imbalance (84/356 categories with ≤5 samples) which will require careful handling in training strategy.

## Experiment 2 — Check environment and install dependencies
- Status: EXPLORE
- Hypothesis: Environment check to verify package versions and data availability
- Change: Created environment check script to assess current setup
- Result: All critical packages (ultralytics, torch, torchvision, timm) are missing; data directories are present and properly structured
- Side effects: No GPU available which will make training extremely slow
- Takeaway: Need to install all required packages before proceeding with any training experiments.

## Experiment 3 — Install dependencies and verify GPU
- Status: FAILED
- Hypothesis: Installing torch with CUDA support will reveal GPU availability; all required packages can be installed
- Change: Modified main.py to install required dependencies with specific versions and verify GPU availability
- Result: packages_installed=0.0, gpu_available=1.0, setup_success=0.0 (torch/torchvision installation failed due to incorrect pip syntax)
- Side effects: Some packages (ultralytics, timm, pycocotools, ensemble-boxes) installed successfully; existing torch 2.10.0+cu128 already available
- Takeaway: The pip install command syntax was incorrect for specifying index URLs - need to use separate --index-url flag, not inline with package specification.

## Experiment 4 — Create train/val split, evaluation function, and YOLO dataset
- Status: FAILED
- Hypothesis: None (refactor experiment)
- Change: Implemented train/val split (224/24), evaluation function with COCOeval, and YOLO format conversion with dataset YAML configs
- Result: ValueError when creating YOLO dataset YAML - path resolution issue with relative_to() method
- Side effects: Successfully created train/val splits and converted annotations to YOLO format before failure
- Takeaway: The YOLO dataset YAML creation failed due to incorrect path handling; need to fix relative path calculation or use absolute paths instead.

## Experiment 5 — Baseline: YOLOv8m nc=1 at 640
- Status: FAILED
- Hypothesis: A single-class YOLOv8m detector at 640px establishes a solid detection baseline, achieving detection_mAP@0.5 > 0.3
- Change: Implemented YOLOv8m nc=1 baseline training with 80 epochs, batch size 16, image size 640
- Result: Training failed due to PyTorch 2.6 weights_only=True security restriction when loading YOLOv8m pretrained weights
- Side effects: None (training never started)
- Takeaway: Need to either set weights_only=False in torch.load or use torch.serialization.add_safe_globals to allowlist ultralytics classes for YOLOv8 pretrained weight loading.

## Experiment 6 — YOLOv8m nc=1 at 1280
- Status: FAILED
- Hypothesis: Higher resolution significantly improves detection on dense shelves with small products, gaining 5+ mAP points
- Change: Modified main.py to train YOLOv8m at 1280px resolution with nc=1, reduced batch size to 8, increased epochs to 100
- Result: Training failed due to PyTorch 2.6 weights_only loading error when attempting to load YOLOv8m pretrained weights
- Side effects: None observed due to failure before training started
- Takeaway: Need to fix PyTorch weights loading compatibility issue by setting weights_only=False or using safe_globals before attempting high-resolution training.
