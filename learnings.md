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
