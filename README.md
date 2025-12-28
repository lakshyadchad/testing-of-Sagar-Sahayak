# 🌊 Sagar Sahayak - Testing Module

> **Maritime Security AI System**: Enhancing underwater visibility and detecting threats in real-time

---

## 📋 Table of Contents
- [What is Sagar Sahayak?](#-what-is-sagar-sahayak)
- [How It Works](#-how-it-works)
- [Quick Start Guide](#-quick-start-guide)
- [What You'll Get](#-what-youll-get)
- [The Difference It Makes](#-the-difference-it-makes)
- [How to Contribute](#-how-to-contribute)
- [Greater Purpose](#-greater-purpose)
- [Technical Details](#-technical-details)

---

## 🎯 What is Sagar Sahayak?

**Sagar Sahayak** (Sanskrit: सागर सहायक, meaning "Ocean Helper") is an AI-powered maritime security system that helps protect our waters by:

- 🔍 **Detecting underwater threats** like mines and suspicious objects
- ✨ **Enhancing murky underwater footage** to crystal-clear visibility
- ⚡ **Processing video in real-time** for immediate threat assessment

Think of it as **"X-Ray Vision for Murky Waters"** - turning unclear, foggy underwater footage into sharp, detectable images!

---

## 🔄 How It Works

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐      ┌──────────────┐
│   Murky     │  →   │  Enhancement │  →   │   Contrast  │  →   │   Detection  │
│   Video     │      │     AI       │      │   Booster   │      │     AI       │
│  (Input)    │      │   (U-Net)    │      │   (CLAHE)   │      │   (YOLO)     │
└─────────────┘      └──────────────┘      └─────────────┘      └──────────────┘
     ❓                     ✨                    🔆                   🎯
  Unclear             Makes Clear           Removes Haze         Finds Objects
```

### The Two-Model Pipeline:

1. **🎨 Enhancement Model** (U-Net Generator)
   - Clears up foggy/murky underwater footage
   - Removes water distortion and particles
   - Works like a "digital water filter"

2. **🎯 Detection Model** (YOLO)
   - Identifies and locates threats in the enhanced image
   - Draws bounding boxes around detected objects
   - Shows confidence scores for each detection

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- Webcam or video file

### Step 1: Install Dependencies

```bash
pip install torch torchvision opencv-python ultralytics pillow numpy
```

### Step 2: Configure the System

Edit `run_system.py` and update these lines:

```python
VIDEO_SOURCE = 'example video/sample_video.mp4'  # Or use 0 for webcam
ENHANCER_WEIGHTS = 'enhancement model/best_model.pth'
DETECTOR_WEIGHTS = 'detection model/best.pt'
```

### Step 3: Run the System

```bash
python run_system.py
```

### Step 4: View Results

- A window will open showing **side-by-side comparison**
- Left: Raw input video
- Right: Enhanced + Detected objects
- Press **'q'** to quit

---

## 🤝 How to Contribute

We welcome contributions! Here's how you can make Sagar Sahayak even better:

### 🎯 Priority Areas:

1. **Improve Model Accuracy**
   - Add more training data (underwater imagery)
   - Fine-tune detection thresholds
   - Test with different water conditions (turbid, clear, deep-sea)

2. **Add New Features**
   - [ ] Support for multiple camera feeds
   - [ ] Record enhanced videos to disk
   - [ ] Alert system for detected threats
   - [ ] Web dashboard for remote monitoring
   - [ ] Integration with drone/ROV systems

3. **Optimize Performance**
   - [ ] Reduce processing latency
   - [ ] Optimize for edge devices (Raspberry Pi, Jetson Nano)
   - [ ] Batch processing mode for recorded videos
   - [ ] Multi-GPU support

4. **Better Documentation**
   - [ ] Add more example videos
   - [ ] Create training tutorials
   - [ ] Document model architecture
   - [ ] Add API documentation

```

### 📝 Contribution Guidelines:

- Write clean, commented code
- Test your changes thoroughly
- Update documentation
- Follow Python PEP 8 style guide
- Add sample videos/images if relevant


---

## 🔧 Technical Details

### Project Structure:

```
testing of Sagar Sahayak/
│
├── 📄 run_system.py              # Main execution script
├── 📄 model_architecture.py      # U-Net Generator definition
├── 📄 README.md                  # This file!
│
├── 📁 detection model/
│   └── best.pt                   # YOLO weights (trained detector)
│
├── 📁 enhancement model/
│   └── best_model.pth            # U-Net weights (trained enhancer)
│
└── 📁 example video/
    ├── sample_video.mp4          # Test videos
    ├── sample_video1.mp4
    ├── sample_video2.mp4
    ├── sample_video3 (1).mp4
    └── sample_video4.mp4
```

### Model Architecture:

#### Enhancement Model (U-Net):
- **Type**: Convolutional Neural Network
- **Architecture**: U-Net Generator (Encoder-Decoder)
- **Input**: 256x256 RGB murky image
- **Output**: 256x256 RGB enhanced image
- **Features**: 
  - 5 downsampling layers (encoder)
  - 5 upsampling layers (decoder)
  - Skip connections for detail preservation
  - Instance normalization for stability

#### Detection Model (YOLO):
- **Type**: Object Detection Neural Network
- **Architecture**: YOLOv8 (or similar)
- **Input**: 1024x1024 enhanced image (upscaled)
- **Output**: Bounding boxes + confidence scores
- **Threshold**: 0.25 confidence

#### Post-Processing:
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
  - Removes white haze
  - Boosts local contrast
  - Makes detections more visible


---

## 📞 Support & Contact

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Join community discussions
- **Documentation**: Check the wiki for detailed guides

---

## 📜 License

This project is developed for maritime security research and defense applications.

---

## 🙏 Acknowledgments

- **U-Net Architecture**: Inspired by medical image segmentation research
- **YOLO**: Ultralytics YOLO for object detection
- **OpenCV**: Computer vision foundation
- **PyTorch**: Deep learning framework

---