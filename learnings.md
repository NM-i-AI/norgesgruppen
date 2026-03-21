# Research Learnings


## Experiment 1 — Explore data structure and existing workspace artifacts
- Status: SUCCESS
- Hypothesis: None (exploration)
- Change: Created exploration script to examine data structure, existing workspace artifacts, and analyze annotations.json, metadata.json, and products directory
- Result: Found 248 images with 22,731 annotations across 356 categories; no existing workspace artifacts (models, runs, splits directories missing)
- Side effects: None
- Takeaway: Dataset has severe class imbalance (41 categories with only 1 annotation, top category has 422) and needs train/val splits created before model training can begin.
