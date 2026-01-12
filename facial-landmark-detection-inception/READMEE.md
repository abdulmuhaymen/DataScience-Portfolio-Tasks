# Facial Landmark Detection with Inception V3

A complete deep learning pipeline for detecting 10 key facial landmarks using MediaPipe and Inception V3, designed for virtual try-on applications like glasses fitting.

## 🎯 Project Overview

This project implements an end-to-end facial landmark detection system:

1. **Data Collection**: Downloads 70K face images from Kaggle
2. **Landmark Extraction**: Uses MediaPipe to extract 10 key facial points
3. **Model Training**: Trains Inception V3 to predict landmarks
4. **Inference**: Predicts landmarks on new face images

### Key Facial Landmarks (10 points)
- Around eyes: 133, 33, 159, 263, 362, 386
- Nose bridge: 6, 197
- Face sides: 127, 356

Perfect for fitting virtual glasses, masks, or AR filters!

## 🚀 Features

- ✅ Processes 2,000 face images from Kaggle dataset
- ✅ MediaPipe Face Mesh for accurate landmark extraction
- ✅ Transfer learning with pretrained Inception V3
- ✅ Train/Val/Test split (75%/15%/10%)
- ✅ Model checkpointing and early stopping
- ✅ Visualization of predictions vs ground truth
- ✅ Real-time inference on new images

## 📊 Technical Details

- **Model**: Inception V3 (ImageNet pretrained)
- **Input Size**: 299×299 pixels
- **Output**: 20 values (10 x,y coordinates)
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Training**: 50 epochs with ReduceLROnPlateau scheduler
- **Batch Size**: 16
- **Device**: CUDA-enabled GPU

## 🛠️ Installation

### Prerequisites
```bash
# Core ML libraries
pip install torch torchvision timm

# Computer vision
pip install mediapipe opencv-python-headless

# Data processing
pip install albumentations numpy pandas

# Utilities
pip install kagglehub tqdm matplotlib pillow
```

### Or install all at once:
```bash
pip install torch torchvision timm mediapipe opencv-python-headless albumentations numpy pandas kagglehub tqdm matplotlib pillow
```

## 📁 Dataset

**Source**: [Kaggle - 70,000 Real Faces Dataset](https://www.kaggle.com/datasets/tunguz/70000-real-faces-1)

The script automatically:
- Downloads the dataset using `kagglehub`
- Processes 2,000 images
- Extracts 10 landmarks per face
- Saves annotations to `facial_landmarks_2000.json`

## 🎮 Usage

### Running the Complete Pipeline
```python
# Simply run the main script
python facial_landmark_detector.py
```

This will:
1. Download the dataset (if not cached)
2. Extract facial landmarks using MediaPipe
3. Train the Inception V3 model
4. Evaluate on test set
5. Show sample predictions

### Inference on Your Own Image

After training, update the image path at the bottom:
```python
# Line ~380 in the script
image_path = "/path/to/your/image.png"
predict_and_show_landmarks(model, image_path, device)
```

## 📈 Training Process

### Hyperparameters
- **Epochs**: 50
- **Learning Rate**: 1e-4
- **Batch Size**: 16
- **Image Size**: 299×299
- **Weight Decay**: 1e-5

### Data Split
- Training: 75% (~1,500 images)
- Validation: 15% (~300 images)
- Test: 10% (~200 images)

### Training Output Example
```
📘 Epoch 1 → Train Loss: 1234.5678 | Val Loss: 1100.1234
✅ Best model saved!
📘 Epoch 2 → Train Loss: 980.4567 | Val Loss: 950.2345
✅ Best model saved!
...
📘 Epoch 50 → Train Loss: 125.4567 | Val Loss: 145.8901
🎯 Final Test Loss: 150.2345
```

## 📊 Model Outputs

### Saved Files
- `facial_landmarks_2000.json` - Extracted landmarks (10 points × 2000 images)
- `best_inception_landmark.pth` - Best model weights
- `processed_faces/` - Images with landmark visualizations
- `predicted_landmarks.png` - Inference result on new image

### Visualization
- **Green dots**: Ground truth landmarks
- **Red dots**: Predicted landmarks
- Landmarks are numbered and displayed on processed images

## 🎨 Sample Results

The model predicts landmarks that closely match the ground truth:
```
Test Loss: ~150 MSE (lower is better)
Visual Accuracy: High precision on eye, nose, and ear regions
```

## 🧠 Model Architecture
```
Inception V3 (Pretrained on ImageNet)
    ↓
Feature Extraction Layers (frozen/fine-tuned)
    ↓
Global Average Pooling
    ↓
Fully Connected Layer (2048 → 20)
    ↓
Output: [x1, y1, x2, y2, ..., x10, y10]
```

## 🔧 Customization

### Change Number of Landmarks
```python
# Line ~20
landmark_indices = [133, 33, 159, 263, 362, 386, 6, 197, 127, 356]
# Add or remove indices from MediaPipe's 478 landmarks

# Update NUM_POINTS accordingly
NUM_POINTS = 10  # Change to match your landmark count
```

### Adjust Training Parameters
```python
# Lines ~50-60
IMAGE_SIZE = 299
BATCH_SIZE = 16
NUM_EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-5
```

## 📱 Applications

- 👓 **Virtual Glasses Try-On**: Fit glasses on face photos
- 😷 **Face Mask Fitting**: Position masks accurately
- 💄 **Makeup Simulation**: Apply virtual makeup
- 🎭 **AR Filters**: Snapchat/Instagram-style filters
- 🎮 **Game Avatars**: Face-based character customization

## 🚀 Future Enhancements

- [ ] Increase dataset to 10K+ images
- [ ] Add data augmentation (rotation, flips, color jitter)
- [ ] Try EfficientNet, ResNet, or Vision Transformers
- [ ] Real-time webcam inference
- [ ] Deploy as Gradio/Streamlit web app
- [ ] Multi-face detection support
- [ ] 3D landmark prediction

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
BATCH_SIZE = 8  # Reduce batch size
```

### No Faces Detected
```python
min_detection_confidence=0.3  # Lower confidence threshold
```

### Slow Training
```python
num_workers=2  # Reduce DataLoader workers
```

## 📄 License

MIT License - feel free to use for personal or commercial projects!

## 🙏 Acknowledgments

- **Dataset**: [Tunguz on Kaggle](https://www.kaggle.com/tunguz)
- **MediaPipe**: Google's Face Mesh solution
- **Inception V3**: Original paper by Szegedy et al.
- **PyTorch**: Facebook AI Research
