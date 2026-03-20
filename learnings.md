# Research Learnings


## Experiment 1 — Verify environment and package versions
- Status: EXPLORE
- Hypothesis: Environment verification to check package versions, GPU availability, and data structure
- Change: Added environment verification script and updated requirements.txt with pinned versions
- Result: Environment mostly ready - 2x NVIDIA A800 80GB GPUs available, core packages installed, data directory structure confirmed with 248 training images
- Side effects: Missing optional packages (pycocotools, ensemble-boxes, scikit-learn) that may be needed for advanced features
- Takeaway: Environment is functional for basic training but missing packages should be installed before attempting ensemble methods or COCO evaluation metrics.
