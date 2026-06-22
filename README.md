# AMIA_PUBLIC_CHALLENGE
This repository contains the end-to-end data processing and deep learning pipeline I developed for the AMIA Public Challenge 2026. The system automates the localization of 14 different thoracic abnormalities (e.g., pulmonary nodules, pleural effusion) from high-resolution chest radiographs. Please find the dataset on kaggle.

Performance: Achieved Rank 43 on the private leaderboard using a single-model YOLO architecture. 

# System Architecture & Engineering
Medical imaging datasets present unique challenges, primarily extreme class imbalance and massive image resolutions. I opted for a local, edge-viable approach using YOLO to balance inference speed with spatial accuracy.

  1. The Data Pipeline
* Resolution Handling: Raw radiographs vary wildly in aspect ratio and size. The pipeline dynamically scales bounding box coordinates into normalized YOLO formats while keeping a persistent lookup table to rescale predictions accurately for submission.
* Data Augmentation: Implemented specialized medical augmentations including CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance bone/tissue contrast, along with slight median blurring to simulate noisy portable X-ray environments.
* Stratification: Splitting involved an 85/15 train/val split. Images with "No Finding" (Class 14) were deliberately filtered during bounding box calculation to prevent the model from suppressing rare abnormality predictions.

  2. Model Training
* Architecture: Fine-tuned YOLO26l initialized with pre-trained weights.
* Optimization: Utilized multi-GPU training (device=[0,1]) using Automatic Mixed Precision (AMP) to maintain memory efficiency at a 640x640 input size.
* Early Stopping: Monitored validation mAP@50-95. Training was halted at epoch 125 (patience=15) to prevent over-fitting, utilizing the weights from epoch 110.

# Results & Evaluation
The primary evaluation metric was PASCAL VOC mAP at an Intersection over Union (IoU) > 0.4.

* Validation mAP50: 0.304
* Validation mAP50-95: 0.166
* Inference Speed: ~8.1ms per image (Tesla T4)

# Failure Analysis & Future Work
While the model serves as a strong, fast baseline, my rank of 43 highlights areas where the architecture struggled, providing a clear roadmap for version 2.0:

1. Small Object Detection: The model performed reasonably well on large abnormalities (e.g., Cardiomegaly) but suffered severe recall drops for smaller, localized features like small pulmonary nodules. Next step: Implement Slicing Aided Hyper Inference (SAHI) to maintain high-resolution feature maps during inference.
2. Class Imbalance: Classes with under 50 instances (e.g., Class 1 and Class 4) plateaued at an mAP50-95 of ~0.09. Next step: Replace the standard loss function with Focal Loss and implement severe oversampling (or mixup) specifically tailored for underrepresented classes.
3. Ensembling: This submission was a single-model baseline. Future iterations would utilize a Weighted Boxes Fusion (WBF) ensemble combining YOLO with a transformer-based detector (like DETR) to capture better global context.
