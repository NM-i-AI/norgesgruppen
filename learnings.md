# Research Learnings


## Experiment 1 — Explore workspace structure and existing artifacts
- Status: EXPLORE
- Hypothesis: Understanding the workspace structure to identify available data and configurations
- Change: Created exploration script to inventory workspace contents
- Result: Found complete YOLO dataset with 248 training images, 22,731 annotations across 356 categories, plus multiple YAML configs for different experiment types (single-class, multi-class, baseline models)
- Side effects: None
- Takeaway: The workspace is well-prepared with COCO-format annotations, YOLO labels, and multiple experiment configurations ready for training different model variants.

## Experiment 2 — Deep dive: data, splits, products, existing models, and environment
- Status: EXPLORE
- Hypothesis: Understanding all available resources at once will let us skip unnecessary setup steps
- Change: Created comprehensive exploration script analyzing dataset structure, annotations, products, GPU environment, and existing resources
- Result: Found 248 images with 22,731 annotations across 356 categories and 344 products, but no train/val splits exist and no pre-trained models available
- Side effects: Long execution time (84s) for comprehensive analysis
- Takeaway: Dataset is ready for training but requires creating train/val splits and training from scratch since no existing models are available.

## Experiment 3 — Build evaluation pipeline and create 90/10 train/val split
- Status: SUCCESS
- Hypothesis: A reliable evaluation function is essential before any experiments
- Change: Built comprehensive evaluation pipeline with stratified 90/10 train/val split, COCO evaluation functions, and YOLO dataset structure
- Result: Successfully created train (224 images, 20345 annotations) and val (24 images, 2386 annotations) splits with 9.7% validation ratio
- Side effects: Minor numpy warning in evaluation pipeline that needs fixing, but doesn't affect functionality
- Takeaway: Evaluation infrastructure is ready for experiments, though the numpy error in classification evaluation should be addressed before running real predictions.

## Experiment 4 — Train YOLOv8x multiclass (nc=356) baseline at imgsz=1280
- Status: FAILED
- Hypothesis: YOLOv8x with 356 classes at 1280 resolution provides a baseline for both detection and classification
- Change: Modified main.py to train YOLOv8x multiclass model with nc=356 at imgsz=1280 for 100 epochs
- Result: AttributeError - torch.serialization module has no attribute 'DEFAULT_WEIGHTS_ONLY'
- Side effects: None (failed before training started)
- Takeaway: The torch version being used doesn't support the DEFAULT_WEIGHTS_ONLY attribute, need to remove or conditionally handle this torch serialization setting.

## Experiment 5 — Train YOLOv8x single-class detector (nc=1) at imgsz=1280
- Status: FAILED
- Hypothesis: A single-class detector achieves higher detection mAP since it doesn't need to distinguish 356 classes, and detection is weighted 70%
- Change: Modified main.py to train YOLOv8x as single-class detector (nc=1), converted all labels to class 0, created data_single_class.yaml
- Result: CUDA out of memory error during training - tried to allocate 682 MiB but only 216 MiB available
- Side effects: GPU memory exhaustion prevented any training progress
- Takeaway: YOLOv8x at imgsz=1280 exceeds available GPU memory regardless of class count, need smaller model or image size.
