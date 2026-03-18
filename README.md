# PyTorch U-Net for Binary Mug Segmentation

## Overview

This repository contains a **binary image segmentation** project developed as part of a university machine learning course. The task is to segment **mugs** from RGB images by predicting a binary mask for each input image.

The project implements a **custom U-Net style convolutional neural network in PyTorch** together with a full training and evaluation pipeline.

My main work focused on:
- implementing and configuring the U-Net architecture
- building the training loop
- adding data augmentation for image / mask pairs
- setting up validation, checkpointing, and IoU-based evaluation

Some parts of the original project scaffolding, particularly parts of the testing setup, were provided as part of the course.

---

## Model

The segmentation model is a **custom deeper U-Net** with:

- encoder-decoder structure
- skip connections
- repeated convolutional blocks with batch normalization and ReLU
- single-channel binary mask prediction

The output is passed through a sigmoid activation to produce pixel-wise values in `[0, 1]`, which are then thresholded to obtain a binary segmentation mask.

---

## Training Pipeline

The project includes:

- **Binary Cross-Entropy loss**
- **Adam optimizer**
- **learning-rate scheduling**
- **early stopping**
- **checkpointing**
- **validation using Intersection over Union (IoU)**

The dataset pipeline applies resizing, normalization, and synchronized augmentation to both image and mask, including flips and color jitter.

---

## Example Result

Average IoU: 0.8469

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MPLCONFIGDIR=/tmp/matplotlib
python src/train.py --data_root ./datasets --ckpt_dir ./checkpoints
```




