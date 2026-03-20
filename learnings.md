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
