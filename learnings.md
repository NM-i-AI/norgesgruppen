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
