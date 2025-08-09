# Quick Start Guide - 130° Fisheye Camera Calibration

## 🚀 5-Minute Setup

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Test Installation
```bash
python3 test_calibration.py
```

### 3. Generate Calibration Pattern
```bash
python3 calibration_utils.py generate
# Print the generated pattern from sample_patterns/chessboard_9x6.png
```

## 🎯 Quick Calibration Process

### Step 1: Capture Images (5 minutes)
```bash
python3 imageCapture.py
```

**Quick Tips:**
- Hold printed chessboard in front of fisheye camera
- **Green corners = good detection** - press 's' to save
- Capture **15-20 images minimum**:
  - 5 images: center of view
  - 5 images: edges and corners of fisheye view
  - 5 images: close distance (fill 50% of frame)
  - 5 images: far distance (chessboard is small)

### Step 2: Run Calibration (1 minute)
```bash
python3 fisheye_calibration.py
```

**Expected Output:**
- RMS Error < 1.0 = Good calibration ✅
- Creates: `calibration_results/fisheye_calibration.json`

### Step 3: Validate Results (30 seconds)
```bash
python3 calibration_utils.py validate
python3 calibration_utils.py compare calib_images/calib_image_000.jpg
```

## 🔧 Common Issues & Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| "No chessboard detected" | Improve lighting, keep pattern flat |
| High RMS error (>2.0) | Capture more images, cover all fisheye areas |
| "Need at least 5 valid images" | Capture more images with better detection |

## 📊 Quality Check

**Excellent Calibration:**
- RMS Error: < 0.5 pixels
- 20+ valid images used
- Straight lines look straight in undistorted image

**Good Calibration:**
- RMS Error: 0.5 - 1.0 pixels
- 15+ valid images used
- Minimal barrel distortion remaining

## 🎯 Using Your Calibration

```python
import json
import cv2
import numpy as np

# Load your calibration
with open('calibration_results/fisheye_calibration.json', 'r') as f:
    calib = json.load(f)

K = np.array(calib['camera_matrix'])
D = np.array(calib['distortion_coefficients'])

# Undistort any image
image = cv2.imread('your_fisheye_image.jpg')
h, w = image.shape[:2]

# Create undistortion maps
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (w, h), np.eye(3), balance=0.0)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)

# Apply undistortion
undistorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
cv2.imwrite('undistorted_output.jpg', undistorted)
```

## 🆘 Need Help?

1. **Check README.md** for detailed explanations
2. **Run test_calibration.py** to verify setup
3. **Ensure good lighting** and flat chessboard
4. **Capture more images** if calibration fails

---
**Total time needed: ~10 minutes for complete calibration process**