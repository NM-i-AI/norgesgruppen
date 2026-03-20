# Research Learnings


## Experiment 1 — Explore dataset structure and statistics
- Status: SUCCESS
- Hypothesis: None (exploration)
- Change: Implemented comprehensive dataset analysis script examining annotations, images, product metadata, and statistics
- Result: Dataset contains 22,731 annotations across 1,000 images with 329 product categories, highly imbalanced distribution (top 10 categories have 50%+ annotations), wide variety of image sizes (720x960 to 5712x4284), and no overlap between annotated product codes and available reference images
- Side effects: None
- Takeaway: The dataset is highly imbalanced with unknown products being a significant category (422 annotations), requiring careful handling of class imbalance and potential data augmentation strategies.

## Experiment 2 — Baseline: Single-class YOLOv8m detection
- Status: SUCCESS
- Hypothesis: A YOLOv8m model trained for single-class detection at image size 1280 can achieve detection_mAP@0.5 > 0.5, giving final_score > 0.35
- Change: Implemented YOLOv8m baseline with COCO to YOLO format conversion, 90/10 train/val split, 30 epochs training at imgsz=1280 with max_det=300
- Result: detection_mAP@0.5 = 0.9726, final_score = 0.6808 (far exceeding expectations)
- Side effects: Training completed in 0.062 hours (~3.7 minutes), generated 361 predictions on 5 validation images
- Takeaway: YOLOv8m achieves excellent detection performance (97.3% mAP@0.5) on this dataset, establishing a strong baseline that significantly exceeds the minimum threshold.

## Experiment 3 — Multi-class YOLOv8m detection (all 329 categories)
- Status: SUCCESS
- Hypothesis: Training YOLOv8m with all categories directly achieves both detection and classification, yielding final_score > single-class approach
- Change: Implemented multi-class YOLOv8m with all 329 categories, custom evaluation metrics (detection_mAP@0.5 + classification_mAP@0.5), and proper category ID mapping
- Result: final_score 0.6808 → 0.7862 (+15.5% improvement)
- Side effects: Longer training time (547.7s), lower individual YOLO metrics (mAP@0.5: 0.5957), but higher task-specific metrics
- Takeaway: Multi-class approach successfully outperforms single-class baseline by directly learning both detection and classification in one model.
