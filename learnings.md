# Research Learnings


## Experiment 1 — Explore data structure and existing workspace artifacts
- Status: SUCCESS
- Hypothesis: None (exploration)
- Change: Created exploration script to examine data structure, existing workspace artifacts, and analyze annotations.json, metadata.json, and products directory
- Result: Found 248 images with 22,731 annotations across 356 categories; no existing workspace artifacts (models, runs, splits directories missing)
- Side effects: None
- Takeaway: Dataset has severe class imbalance (41 categories with only 1 annotation, top category has 422) and needs train/val splits created before model training can begin.

## Experiment 2 — Inspect annotation details, categories, and product reference images
- Status: EXPLORE
- Hypothesis: Understanding dataset structure and characteristics to inform model design decisions
- Change: Created comprehensive analysis script examining annotations, categories, metadata, and reference images
- Result: Dataset has 22,731 annotations across 356 categories with heavy class imbalance; 84.2% are large objects (≥96²); most products have multi-view reference images (avg 4.6 per product)
- Side effects: None (analysis only)
- Takeaway: The severe class imbalance (top category has 422 annotations vs 41 categories with only 1) and abundance of reference images suggest a few-shot learning approach could be more effective than traditional classification.

## Experiment 3 — Create train/val split, YOLO labels, evaluation function, and data.yaml
- Status: SUCCESS
- Hypothesis: None (refactor)
- Change: Implemented complete data preparation pipeline with 90/10 stratified split, COCO/YOLO format conversion, and evaluation function
- Result: Successfully created 224 train/24 val images with 20540/2191 annotations across 356 categories
- Side effects: Evaluation function has warnings with dummy data but works correctly (returns 0.0000 as expected)
- Takeaway: Data preparation infrastructure is complete and ready for model training with both multi-class and single-class variants available.

## Experiment 4 — Baseline: YOLOv8m multi-class at 640
- Status: FAILED
- Hypothesis: YOLOv8m with 356 classes at 640px provides a reasonable starting baseline
- Change: Implemented complete YOLOv8m training pipeline with 356 classes, 640px resolution, 80 epochs, batch size 16
- Result: Training failed due to PyTorch 2.6 weights_only=True security restriction when loading pretrained weights
- Side effects: Model architecture loaded successfully (26M parameters, 80.2 GFLOPs) before failing at weight loading
- Takeaway: Need to handle PyTorch 2.6's new weights_only=True default by either setting weights_only=False or using safe_globals context manager for YOLO model loading.

## Experiment 5 — YOLOv8x multi-class at 1280
- Status: FAILED
- Hypothesis: Larger model + higher resolution will improve both detection and classification on shelf images
- Change: Upgraded from YOLOv8m to YOLOv8x, increased resolution from 640px to 1280px, added auto-batch sizing and AMP
- Result: Training failed due to PyTorch 2.6 weights_only security restriction when loading pretrained YOLOv8x weights
- Side effects: Model successfully loaded (68M parameters, 260 GFLOPs) but crashed during weight transfer from pretrained checkpoint
- Takeaway: Need to add torch.serialization.safe_globals or set weights_only=False to load YOLOv8x pretrained weights in PyTorch 2.6+.
