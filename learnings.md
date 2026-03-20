# Research Learnings

## Experiment 1 — Explore dataset structure and statistics
- Status: EXPLORE
- Hypothesis: Understanding dataset structure to inform model architecture choices
- Change: Enhanced exploration script to analyze product codes, image distributions, and file paths
- Result: Discovered 248 training images with 22,731 annotations across 356 categories, severe class imbalance (74 categories <5 examples), and 0% manually corrected annotations
- Side effects: Script crashed on division by zero when cross-referencing product codes (0 annotated products found)
- Takeaway: Dataset has extreme class imbalance and dense annotations (avg 91.7 per image) suggesting need for robust detection model and careful sampling strategy for rare categories.

## Experiment 2 — Single-class YOLOv8m detection baseline at 1280px
- Status: FAILED
- Hypothesis: A single-class YOLOv8m detector at 1280px trained for 50 epochs will achieve detection_mAP@0.5 > 0.60, giving final_score > 0.42 (with classification=0)
- Change: Simplified to single-class detection with YOLOv8m, reduced epochs to 50, increased batch size to 8, and mapped all categories to class 0
- Result: Import error - pycocotools module not found, preventing execution
- Side effects: None observed due to early failure
- Takeaway: The pycocotools dependency needs to be installed before the experiment can run.

## Experiment 3 — Multi-class YOLOv8m detection at 1280px
- Status: FAILED
- Hypothesis: Training with all 356 categories enables both detection and classification scoring, achieving final_score > 0.50 (0.7*det + 0.3*cls)
- Change: Modified baseline to train with all 356 categories, added classification evaluation, increased epochs to 80, reduced batch size to 6
- Result: Import error - missing pycocotools dependency
- Side effects: None (failed before execution)
- Takeaway: The pycocotools library needs to be installed before running multi-class experiments that use COCO evaluation functions.
