# Chest X-Ray Pneumonia Classification with GoogLeNet

A PyTorch implementation of binary pneumonia detection using a custom **GoogLeNet (Inception v1)** architecture with auxiliary classifiers on chest X-ray images.

## Overview

This project implements a deep learning model to classify chest X-ray images as either **NORMAL** or **PNEUMONIA** using a faithful reproduction of the GoogLeNet (Inception v1) architecture. The implementation includes auxiliary classifiers, data augmentation, and class balancing to achieve robust pneumonia detection.

## Dataset

The project uses the [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) from Kaggle, containing 5,863 X-ray images organized into train, validation, and test sets with two classes:

- **NORMAL**: Healthy chest X-rays  
- **PNEUMONIA**: X-rays showing bacterial or viral pneumonia

## Features

- **Custom GoogLeNet Architecture**: Complete implementation of Inception v1 with multi-scale feature extraction  
- **Auxiliary Classifiers**: Two auxiliary outputs during training to combat vanishing gradients  
- **Data Augmentation**: Random flips, rotations, and brightness adjustments  
- **Class Balancing**: Weighted loss function to handle dataset imbalance  
- **Model Checkpointing**: Saves best model based on validation performance  
- **Comprehensive Metrics**: Confusion matrix, per-class accuracy, and classification report  

## Architecture Details

### GoogLeNet Components

- **Initial Convolutions**: 7×7 and 3×3 convolutions with local response normalization  
- **Inception Modules**: 9 inception blocks with parallel 1×1, 3×3, and 5×5 convolutions  
- **Auxiliary Classifiers**: Two auxiliary branches for intermediate supervision (weighted 0.3)  
- **Global Average Pooling**: Reduces spatial dimensions before final classification  
- **Dropout Regularization**: 0.5 dropout rate in auxiliary and main classifiers  

### Data Pipeline

- **Input Size**: 224×224 RGB images  
- **Training Augmentation**: Horizontal flip, rotation (±15°), brightness jitter (1.0–1.3)  
- **Normalization**: ImageNet mean and standard deviation  
- **Batch Size**: 32 images per batch  

### Training Configuration

- **Optimizer**: AdamW with learning rate 1e-4 and weight decay 1e-4  
- **Loss Function**: CrossEntropyLoss with computed class weights  
- **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)  
- **Epochs**: 25 with early stopping based on validation loss  

### Key Functions

- `build_df()`: Creates DataFrame with image paths and labels from directory structure  
- `ChestXrayDataset`: Custom Dataset class handling image loading and transforms  
- `InceptionA/InceptionB`: Inception module implementations with parallel convolutions  
- `AuxClassifier`: Auxiliary classifier branches for intermediate supervision  
- `GoogLeNetCustom`: Main model class with forward pass handling auxiliary outputs  

###  Output

### The implementation provides:

- Dataset Statistics: Training, validation, and test set sizes, class distribution, and computed class weights

- Training Progress: Per-epoch loss/accuracy, progress bars, and checkpoint saving

- Evaluation Metrics: Overall test accuracy, per-class accuracy, confusion matrix, and classification report

- Visualizations: Sample batch of training images, confusion matrix heatmap, and training/validation metrics over epochs

### Performance

- The model achieves pneumonia detection through:

- Multi-scale feature extraction via inception modules

- Auxiliary loss supervision preventing vanishing gradients

- Class-weighted loss handling dataset imbalance (~3:1 PNEUMONIA:NORMAL ratio)

- Data augmentation improving generalization