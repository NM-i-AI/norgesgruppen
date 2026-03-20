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
