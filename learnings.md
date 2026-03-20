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
