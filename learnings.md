# Research Learnings


## Experiment 1 — Explore dataset structure and statistics
- Status: EXPLORE
- Metrics: {}
- Logger error: Error code: 400 - {'error': {'message': 'anthropic/claude-haiku is not a valid model ID', 'code': 400}, 'user_id': 'user_3AigRyjDp4NrZP6HahtpV7bmm55'}


## Experiment 2 — YOLOv8m single-class detection baseline at 1280px
- Status: FAILED
- Metrics: {}
- Logger error: Error code: 400 - {'error': {'message': 'anthropic/claude-haiku is not a valid model ID', 'code': 400}, 'user_id': 'user_3AigRyjDp4NrZP6HahtpV7bmm55'}


## Experiment 1 — Explore dataset structure and statistics
- Status: EXPLORE
- Hypothesis: Understanding dataset structure to inform model architecture choices
- Change: Enhanced exploration script to analyze product codes, image distributions, and file paths
- Result: Discovered 248 training images with 22,731 annotations across 356 categories, severe class imbalance (74 categories <5 examples), and 0% manually corrected annotations
- Side effects: Script crashed on division by zero when cross-referencing product codes (0 annotated products found)
- Takeaway: Dataset has extreme class imbalance and dense annotations (avg 91.7 per image) suggesting need for robust detection model and careful sampling strategy for rare categories.
