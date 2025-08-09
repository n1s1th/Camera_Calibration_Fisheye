# Camera_Calibration_Fisheye
# Camera_Calibration_Fisheye

A comprehensive step-by-step guide and toolkit for calibrating wide-angle 130-degree fisheye cameras using OpenCV. This project provides a robust pipeline for fisheye camera calibration, distortion correction, and validation.

## 🎯 Overview

Fisheye cameras with 130-degree field of view introduce significant barrel distortion that needs to be corrected for computer vision applications. This toolkit provides everything you need to calibrate your fisheye camera and obtain accurate calibration parameters.

## 📋 Prerequisites

- Raspberry Pi with camera module (or any computer with camera)
- Python 3.7 or higher
- Printed chessboard calibration pattern (recommended: 9x6 corners, 25mm squares)
- Good lighting conditions

## 🛠 Installation

1. Clone this repository:
```bash
git clone https://github.com/n1s1th/Camera_Calibration_Fisheye.git
cd Camera_Calibration_Fisheye
```

2. Install required dependencies:
```bash
pip3 install -r requirements.txt
```

## 📖 Step-by-Step Calibration Guide

### Step 1: Prepare Calibration Pattern

1. **Download and print a chessboard pattern**:
   - Use a 9x6 chessboard (9 corners horizontally, 6 corners vertically)
   - Print on A4 paper with 25mm square size
   - Mount on rigid surface (cardboard/clipboard) to keep it flat

2. **Pattern Requirements**:
   - High contrast black and white squares
   - Perfectly flat surface
   - No wrinkles or bends
   - Clean edges and corners

### Step 2: Capture Calibration Images

1. **Run the image capture script**:
```bash
python3 imageCapture.py
```

2. **Capture Strategy for 130-degree Fisheye**:
   - **Minimum 15-20 images** (more is better - aim for 30+)
   - Position chessboard at **different distances**: close, medium, far
   - Cover **all areas** of the fisheye field of view:
     - Center of image
     - Top, bottom, left, right edges
     - All four corners (important for fisheye!)
   - **Various orientations**: horizontal, vertical, diagonal
   - **Different tilting angles**: 0°, 15°, 30°, 45°

3. **Capture Tips**:
   - Green corners indicate chessboard detection ✅
   - Only save images when chessboard is detected
   - Ensure good lighting (avoid shadows)
   - Keep chessboard steady when saving
   - Press 's' to save, 'q' to quit

### Step 3: Run Fisheye Calibration

1. **Start calibration process**:
```bash
python3 fisheye_calibration.py
```

2. **What happens during calibration**:
   - Detects chessboard corners in all images
   - Calculates fisheye distortion parameters
   - Computes camera intrinsic matrix
   - Outputs calibration quality metrics

3. **Expected output**:
   - RMS error < 1.0 pixel (excellent), < 2.0 (good)
   - Camera matrix (K) with focal lengths and optical center
   - Distortion coefficients (k1, k2, k3, k4)
   - Results saved to `calibration_results/fisheye_calibration.json`

### Step 4: Validate Calibration Results

1. **Check calibration quality**:
   - Low RMS error indicates good calibration
   - Visual inspection of undistorted test image
   - Straight lines should appear straight in undistorted image

2. **Test undistortion**:
```bash
python3 -c "
from fisheye_calibration import FisheyeCalibrator
calibrator = FisheyeCalibrator()
calibrator.undistort_image('calib_images/calib_image_000.jpg')
print('Check calibration_results/ for undistorted image')
"
```

## 🔧 Advanced Usage

### Custom Chessboard Size

If using different chessboard pattern:

```python
from fisheye_calibration import FisheyeCalibrator

# For 7x5 chessboard with 30mm squares
calibrator = FisheyeCalibrator(chessboard_size=(7, 5), square_size=30.0)
results = calibrator.calibrate_fisheye()
```

### Batch Image Undistortion

```python
import glob
from fisheye_calibration import FisheyeCalibrator

calibrator = FisheyeCalibrator()
for image_path in glob.glob("test_images/*.jpg"):
    calibrator.undistort_image(image_path)
```

### Using Calibration Data in Your Application

```python
import json
import cv2
import numpy as np

# Load calibration results
with open('calibration_results/fisheye_calibration.json', 'r') as f:
    calib_data = json.load(f)

K = np.array(calib_data['camera_matrix'])
D = np.array(calib_data['distortion_coefficients'])

# Use in your application for real-time undistortion
# ... your camera loop ...
undistorted = cv2.fisheye.undistortImage(image, K, D)
```

## 📊 Understanding Calibration Results

### Camera Matrix (K)
```
[[fx  0  cx]
 [0  fy  cy]
 [0   0   1]]
```
- `fx, fy`: Focal lengths in pixels
- `cx, cy`: Principal point (optical center)

### Distortion Coefficients (D)
- `[k1, k2, k3, k4]`: Fisheye distortion parameters
- Higher values indicate more distortion

### Quality Metrics
- **RMS Error**: 
  - < 0.5: Excellent calibration
  - 0.5-1.0: Good calibration
  - 1.0-2.0: Acceptable calibration
  - > 2.0: Poor calibration (recalibrate)

## 🔍 Troubleshooting

### Common Issues and Solutions

1. **"No chessboard detected"**:
   - Ensure good lighting
   - Check chessboard is flat and unobstructed
   - Verify correct chessboard size in code
   - Try different distances and angles

2. **High RMS error (>2.0)**:
   - Capture more images (30+ recommended)
   - Cover all areas of fisheye field of view
   - Ensure chessboard images are sharp (not blurry)
   - Check for consistent chessboard pattern

3. **"Need at least 5 valid images"**:
   - More images were rejected than accepted
   - Improve image quality and chessboard detection
   - Check lighting and focus

4. **Distorted results after undistortion**:
   - May need to adjust balance parameter in undistortion
   - Try different regions of interest (ROI)
   - Verify calibration with more test images

## 📁 File Structure

```
Camera_Calibration_Fisheye/
├── README.md                    # This guide
├── requirements.txt             # Python dependencies
├── imageCapture.py             # Enhanced image capture with preview
├── fisheye_calibration.py      # Main calibration script
├── calib_images/               # Captured calibration images
├── calibration_results/        # Calibration outputs
│   ├── fisheye_calibration.json
│   └── *_undistorted.jpg
└── sample_patterns/            # Sample calibration patterns
```

## 🎓 Tips for Best Results

1. **Image Quality**: Use high resolution, sharp, well-lit images
2. **Pattern Coverage**: Ensure chessboard appears in all regions of fisheye view
3. **Quantity**: More images generally lead to better calibration (30+ recommended)
4. **Variety**: Different positions, orientations, and distances
5. **Stability**: Keep camera and chessboard steady during capture
6. **Validation**: Always test calibration on new images

## 📚 References

- [OpenCV Fisheye Camera Calibration](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html)
- [Camera Calibration Theory](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available under the MIT License.
