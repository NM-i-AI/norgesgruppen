# Research Learnings


## Experiment 1 — Explore workspace structure and existing artifacts
- Status: SUCCESS
- Hypothesis: None (exploration)
- Change: Created exploration script to analyze workspace structure, data directory contents, annotations.json statistics, metadata, and existing artifacts
- Result: Successfully mapped complete dataset: 248 images, 22,731 annotations across 356 product categories, with train/val splits already prepared
- Side effects: None
- Takeaway: Dataset is ready for training with 356 grocery product categories, high annotation density (91.7 per image), and existing train/val splits in place.

## Experiment 2 — Verify environment, GPU, packages, and existing artifacts
- Status: EXPLORE
- Hypothesis: Environment verification to understand available resources and constraints
- Change: Added comprehensive environment verification script checking Python, GPU, packages, data, and system resources
- Result: Environment fully mapped - RTX 4090 GPU available, 503GB RAM, 210 training images with 22731 boxes across 356 categories, but ultralytics model loading fails due to PyTorch 2.6 weights_only security change
- Side effects: Discovered critical compatibility issue with ultralytics and PyTorch 2.6 that will block training
- Takeaway: Must resolve PyTorch weights_only compatibility issue before proceeding with any YOLO training experiments.

## Experiment 3 — Fix PyTorch 2.6 / ultralytics compatibility and verify YOLO training works
- Status: SUCCESS
- Hypothesis: Monkey-patching torch.load to use weights_only=False or upgrading ultralytics will fix the model loading issue, enabling YOLO training.
- Change: Added monkey-patch for torch.load with weights_only=False, comprehensive YOLO training test, and pyyaml dependency
- Result: model_loading_success=1.0, test_training_success=1.0 (both metrics achieved for first time)
- Side effects: Training completed quickly (0.007 hours for 1 epoch), validation ran successfully with 18.99it/s processing speed
- Takeaway: The PyTorch 2.6 / ultralytics compatibility issue is fully resolved and YOLO training pipeline is now functional.

## Experiment 4 — Build evaluation function and prepare YOLO dataset YAML
- Status: SUCCESS
- Hypothesis: A correct evaluation function matching the competition metric (0.7*detection_mAP@0.5 + 0.3*classification_mAP@0.5) is critical for reliable experiment comparison.
- Change: Built comprehensive evaluation function using pycocotools that computes detection_mAP@0.5 (category-agnostic) and classification_mAP@0.5 (category-specific), then combines them with competition formula; added YOLO dataset preparation and train/val split utilities
- Result: All functions working correctly - evaluation_function_success=1.0, yolo_dataset_success=1.0, dummy validation produces expected low scores (val_score=0.0277, detection_mAP=0.0295, classification_mAP=0.0234)
- Side effects: Added pycocotools dependency, evaluation takes ~10 seconds due to COCO API overhead
- Takeaway: Core evaluation infrastructure is ready and validated - can now proceed with baseline model training and systematic experiments.

## Experiment 5 — Baseline: YOLOv8x multiclass (nc=356) at 1280px, 50 epochs
- Status: FAILED
- Hypothesis: YOLOv8x at 1280px gives a strong multiclass baseline with 50 epochs for quick feedback
- Change: Implemented YOLOv8x training with nc=356 classes at 1280px resolution, batch size auto-selected as 4
- Result: Training failed due to CUDA out of memory error during first epoch (0.0 val_score, training_failed=1.0)
- Side effects: Memory usage exceeded 65GB on 80GB GPU, indicating very high memory requirements
- Takeaway: YOLOv8x at 1280px is too memory-intensive even with batch size 4; need to reduce model size or image resolution.

## Experiment 6 — Train single-class detector: YOLOv8x nc=1 at 1280px, 50 epochs
- Status: FAILED
- Hypothesis: A single-class detector should have higher detection recall since it doesn't need to distinguish 356 categories, and detection is weighted 70% in the score.
- Change: Modified main.py to train YOLOv8x with nc=1, mapping all 22,731 annotations to class 0
- Result: Training failed with CUDA device-side assert triggered in TAL (Task-Aligned Learning) module during bbox_scores assignment
- Side effects: Model loaded successfully and batch size was auto-determined as 4, but crashed during first epoch
- Takeaway: The single-class mapping likely created invalid label indices that triggered CUDA assertions in the loss computation, requiring proper label validation before training.

## Experiment 7 — Properly prepare YOLO label files and train YOLOv8m nc=356 at 640px, 30 epochs
- Status: FAILED
- Hypothesis: YOLOv8m at 640px with proper YOLO-format label files will train successfully without OOM, giving us a working multiclass baseline.
- Change: Converted COCO annotations to YOLO format, created proper train/val directory structure with symlinked images, and attempted to train YOLOv8m at 640px for 30 epochs.
- Result: CUDA OOM error during training - tried to allocate 572 MiB but only 460 MiB free, despite AutoBatch suggesting batch size 122.
- Side effects: Dataset preparation was successful (labels cached properly), but AutoBatch calculation appears unreliable for actual training memory requirements.
- Takeaway: Even YOLOv8m with AutoBatch optimization still hits OOM, so we need to either use a smaller model (YOLOv8s/n) or manually set a much smaller batch size.

## Experiment 8 — Fix batch size and category mapping, train YOLOv8m nc=356 at 640px, 30 epochs
- Status: FAILED
- Hypothesis: Setting batch=8 explicitly (instead of AutoBatch) and fixing category_id to 0-indexed YOLO format will allow YOLOv8m training to complete successfully.
- Change: Set batch=8 explicitly, added category mapping from COCO IDs to 0-indexed range, cleaned up stale dataset directory
- Result: Training failed due to numpy compatibility issue - `numpy.trapz` function was removed in newer numpy versions
- Side effects: Model loaded successfully and began training before hitting the numpy error during validation
- Takeaway: The batch size and category mapping fixes worked, but ultralytics has a dependency issue with newer numpy versions that removes the `trapz` function.

## Experiment 9 — Fix numpy.trapz compatibility and train YOLOv8m nc=356 at 640px, 30 epochs
- Status: SUCCESS
- Hypothesis: Adding a monkey-patch for numpy.trapz (using numpy.trapezoid if available, or scipy.integrate.trapezoid) will fix the validation crash, allowing YOLOv8m training to complete successfully.
- Change: Added numpy.trapz compatibility fix at the top of main.py before importing ultralytics
- Result: Training completed successfully with val_score=0.2812 (detection_map=0.3518, classification_map=0.1166)
- Side effects: None observed - training completed in 272.7s with 1177 predictions on 24 validation images
- Takeaway: The numpy.trapz compatibility fix successfully resolved the validation crash, enabling full YOLOv8m training completion with reasonable performance metrics.
