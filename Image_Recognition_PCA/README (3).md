# Image Recognition with PCA (Principal Component Analysis)

A Python implementation of face recognition using Principal Component Analysis (PCA) and eigenfaces on the Avengers Faces Dataset.

## Overview

This project demonstrates facial recognition using PCA dimensionality reduction and a simple nearest neighbor classifier. The implementation includes eigenface computation, image projection into PCA space, and classification with performance evaluation.

## Dataset

The project uses the [Avengers Faces Dataset](https://www.kaggle.com/datasets/yasserh/avengers-faces-dataset) from Kaggle, which contains images of Avengers characters organized into train, validation, and test sets.

## Features

- **PCA Implementation**: Custom PCA computation using eigenface methodology
- **Image Preprocessing**: Grayscale conversion and resizing to 100x100 pixels
- **Dimensionality Reduction**: Projects high-dimensional image data into lower-dimensional PCA space
- **Classification**: Nearest neighbor classifier in PCA space
- **Visualization**: Displays mean face, eigenfaces, and reconstruction examples
- **Performance Metrics**: Accuracy scores and detailed classification reports



## Usage

1. **Download Dataset**: The script automatically downloads the Avengers faces dataset using `kagglehub`


### Data Loading
- Downloads dataset from Kaggle Hub
- Loads images from train/val/test splits
- Converts images to grayscale and resizes to 100x100
- Creates label mappings for character classes

### PCA Implementation
- Computes mean face from training data
- Calculates eigenfaces using covariance matrix eigendecomposition
- Reduces dimensionality to top k=100 principal components

### Classification
- Projects all datasets into PCA space
- Uses nearest neighbor classification
- Evaluates performance on validation and test sets

### Visualization
- Displays mean face and top 5 eigenfaces
- Shows original vs reconstructed images
- Provides classification reports

## Key Functions

- `load_dataset()`: Loads and preprocesses images from a directory
- `compute_pca()`: Computes PCA eigenfaces and eigenvalues
- `project()`: Projects images into PCA space
- `nearest_neighbor()`: Performs classification using nearest neighbor
- `reconstruct()`: Reconstructs images from PCA projections

## Parameters

- `IMG_SIZE`: Image dimensions (100x100 pixels)
- `k`: Number of principal components (100)
- Dataset automatically splits into train/validation/test

## Output

The script provides:
- Dataset statistics (number of images per class)
- Eigenface visualizations
- Validation and test accuracy scores
- Detailed classification report with precision, recall, and F1-scores
- Original vs reconstructed image comparisons

## Performance

The model achieves face recognition through dimensionality reduction from 10,000 dimensions (100x100 pixels) to 100 principal components, maintaining classification accuracy while significantly reducing computational complexity.

## Technical Details

- Uses eigenface methodology for PCA computation
- Implements nearest neighbor classification in reduced dimensional space
- Handles grayscale image processing
- Provides comprehensive error handling for image loading
- Includes visualization capabilities for model interpretation

