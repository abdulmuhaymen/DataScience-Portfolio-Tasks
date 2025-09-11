# 🖼️ Image Stitching Project

A comprehensive image stitching application that creates panoramic images from multiple overlapping photographs using advanced computer vision techniques. The project includes both a core Python implementation and a user-friendly Streamlit web interface.

## 🌟 Features

- **Dense SIFT Feature Extraction**: Robust feature detection using dense SIFT descriptors
- **Automatic Image Alignment**: Pairwise homography computation with RANSAC outlier rejection
- **Smart Center Image Selection**: Automatically selects the best center image for optimal stitching
- **Feathered Blending**: Seamless image blending with distance-based weight maps
- **Interactive Web Interface**: Easy-to-use Streamlit app for non-technical users
- **Pickle Support**: Save and load stitching results for later use
- **Progress Tracking**: Real-time progress updates during processing


```


```



## 📖 Usage

### Web Interface Usage

1. **Choose Mode**: Select from "Stitch New Images", "Load from Pickle", or "Both"
2. **Upload Images**: Upload 2+ overlapping images (JPG, PNG supported)
3. **Adjust Parameters**: Configure max dimension and SIFT step size
4. **Process**: Click "Start Stitching" and wait for completion
5. **Download**: Save results as JPG or pickle file



## 🔬 Algorithm Overview

### 1. **Feature Extraction**
- Uses dense SIFT (Scale-Invariant Feature Transform) descriptors
- Extracts features on a regular grid for comprehensive coverage
- Configurable step size for density control

### 2. **Feature Matching**
- Brute-force matcher with L2 norm
- Lowe's ratio test (default: 0.75) for robust matching
- Cross-validation for match quality

### 3. **Homography Estimation**
- RANSAC-based outlier rejection
- Minimum 8 matches required for homography computation
- Configurable inlier threshold (default: 4.0 pixels)

### 4. **Image Registration**
- Graph-based approach to find optimal transformation tree
- Automatic center image selection based on connectivity
- Breadth-first search for global transformation computation

### 5. **Image Blending**
- Distance transform-based weight computation
- Feathered blending for seamless transitions
- Multi-band blending support for improved quality

## 🌐 Web Interface

### Features
- **Drag & Drop Upload**: Easy image uploading
- **Real-time Preview**: See uploaded images before processing
- **Progress Tracking**: Visual progress bars and status updates
- **Parameter Control**: Adjust stitching parameters via sliders
- **Result Visualization**: Interactive result display
- **Download Options**: Multiple export formats

### Supported Formats
- **Input**: JPG, JPEG, PNG
- **Output**: JPG (high quality), PKL (pickle format)

## ⚙️ Parameters

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_DIM` | 1600 | Maximum image dimension (pixels) |
| `step_size` | 8 | SIFT descriptor step size |
| `LOWE_RATIO` | 0.75 | Lowe's ratio test threshold |
| `RANSAC_THRESH` | 4.0 | RANSAC inlier threshold (pixels) |
| `MIN_INLIERS` | 30 | Minimum inliers for valid homography |

### Performance Tuning

- **Reduce `MAX_DIM`** for faster processing
- **Increase `step_size`** for fewer features and faster matching
- **Adjust `LOWE_RATIO`** for match quality vs. quantity trade-off

