# Research Learnings


## Experiment 1 — Explore dataset structure and annotations
- Status: SUCCESS
- Hypothesis: None (exploration)
- Change: Implemented comprehensive dataset exploration script to understand COCO annotations, metadata, and reference images
- Result: Dataset contains 248 images with 22,731 annotations across 356 product categories; severe class imbalance with 84 categories having ≤5 annotations
- Side effects: None
- Takeaway: The dataset has extreme class imbalance (84/356 categories with ≤5 samples) which will require careful handling in training strategy.
