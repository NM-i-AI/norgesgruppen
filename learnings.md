# Research Learnings


## Experiment 1 — Verify environment and package versions
- Status: EXPLORE
- Hypothesis: Environment verification to check package versions, GPU availability, and data structure
- Change: Added environment verification script and updated requirements.txt with pinned versions
- Result: Environment mostly ready - 2x NVIDIA A800 80GB GPUs available, core packages installed, data directory structure confirmed with 248 training images
- Side effects: Missing optional packages (pycocotools, ensemble-boxes, scikit-learn) that may be needed for advanced features
- Takeaway: Environment is functional for basic training but missing packages should be installed before attempting ensemble methods or COCO evaluation metrics.

## Experiment 2 — Deep dive into dataset structure, annotations, and metadata
- Status: EXPLORE
- Hypothesis: Understanding the full dataset structure will inform model design choices
- Change: Implemented comprehensive dataset analysis examining COCO annotations, image statistics, category distribution, bounding box properties, and product reference images
- Result: Dataset has 248 images with 22,731 annotations across 356 categories (avg 91.7 annotations/image), severe class imbalance with 110 categories having <10 annotations, and 344 product reference images with multiple viewpoints
- Side effects: None (analysis only)
- Takeaway: This is a dense, few-shot detection problem requiring strategies for extreme class imbalance and leveraging multi-view product reference images.

## Experiment 3 — Create train/val split, evaluation function, and YOLO data conversion
- Status: SUCCESS
- Hypothesis: A proper evaluation setup is essential before any training
- Change: Created 90/10 train/val split (223/25 images), COCO-to-YOLO conversion for both multi-class (nc=356) and single-class (nc=1) formats, and evaluation function using pycocotools
- Result: Successfully created training pipeline with 20,236 train and 2,495 val annotations, evaluation function tested and working
- Side effects: Minor numpy warning in evaluation function with dummy predictions, but core functionality works
- Takeaway: Training infrastructure is now ready - can proceed with actual model training using either multi-class or single-class YOLO formats.
