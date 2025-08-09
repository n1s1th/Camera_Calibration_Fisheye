#!/usr/bin/env python3
"""
Fisheye Calibration Usage Example
This script demonstrates how to use calibration results in your application
"""

import cv2
import numpy as np
import json
import os

def load_calibration_data(calibration_file="calibration_results/fisheye_calibration.json"):
    """Load calibration data from JSON file"""
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
    
    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)
    
    K = np.array(calib_data['camera_matrix'])
    D = np.array(calib_data['distortion_coefficients'])
    
    return K, D, calib_data

def create_undistortion_maps(K, D, image_size, balance=0.0):
    """Create undistortion maps for efficient real-time processing"""
    w, h = image_size
    
    # Estimate new camera matrix
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance)
    
    # Create undistortion maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    
    return map1, map2, new_K

def undistort_image_fast(image, map1, map2):
    """Fast image undistortion using precomputed maps"""
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)

def example_real_time_undistortion():
    """Example of real-time fisheye undistortion setup"""
    print("=== Real-time Fisheye Undistortion Example ===")
    
    try:
        # Load calibration
        K, D, calib_data = load_calibration_data()
        print(f"✓ Loaded calibration data (RMS: {calib_data['rms_error']:.3f})")
        
        # Get image dimensions from calibration
        image_size = calib_data['image_size']
        print(f"✓ Image size: {image_size}")
        
        # Create undistortion maps (do this once, reuse for all frames)
        map1, map2, new_K = create_undistortion_maps(K, D, image_size)
        print("✓ Created undistortion maps")
        
        print(f"""
Real-time Processing Setup Complete!

To use in your camera loop:

```python
# Initialize once
K, D, calib_data = load_calibration_data()
image_size = calib_data['image_size']
map1, map2, new_K = create_undistortion_maps(K, D, image_size)

# In your camera loop
while True:
    frame = camera.capture_array()  # Your camera capture
    undistorted_frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    # Use undistorted_frame for processing
```
        """)
        
        return True
        
    except FileNotFoundError:
        print("✗ No calibration data found. Run fisheye_calibration.py first.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def example_batch_processing():
    """Example of batch image processing"""
    print("\n=== Batch Image Processing Example ===")
    
    try:
        # Load calibration
        K, D, calib_data = load_calibration_data()
        
        # Example with calibration images if they exist
        import glob
        images = glob.glob("calib_images/*.jpg")
        
        if not images:
            print("No images found for demonstration")
            return
        
        print(f"Processing {len(images)} images...")
        
        for i, image_path in enumerate(images[:3]):  # Process first 3 for demo
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            # Create maps for this image size
            map1, map2, new_K = create_undistortion_maps(K, D, (w, h))
            
            # Undistort
            undistorted = undistort_image_fast(img, map1, map2)
            
            # Save result
            output_path = f"calibration_results/demo_undistorted_{i:02d}.jpg"
            os.makedirs("calibration_results", exist_ok=True)
            cv2.imwrite(output_path, undistorted)
            print(f"  ✓ Processed: {os.path.basename(image_path)} -> {output_path}")
        
        print("Batch processing complete!")
        
    except Exception as e:
        print(f"✗ Batch processing error: {e}")

def print_calibration_summary():
    """Print a summary of calibration results"""
    print("\n=== Calibration Summary ===")
    
    try:
        K, D, calib_data = load_calibration_data()
        
        print(f"RMS Error: {calib_data['rms_error']:.4f} pixels")
        print(f"Images Used: {calib_data['valid_images']}/{calib_data['total_images']}")
        print(f"Image Resolution: {calib_data['image_size'][0]}x{calib_data['image_size'][1]}")
        
        # Camera parameters
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        
        print(f"\nCamera Matrix:")
        print(f"  Focal lengths: fx={fx:.1f}, fy={fy:.1f}")
        print(f"  Principal point: cx={cx:.1f}, cy={cy:.1f}")
        
        print(f"\nDistortion Coefficients:")
        for i, coeff in enumerate(D.flatten()):
            print(f"  k{i+1} = {coeff:.6f}")
        
        # Estimate field of view
        w, h = calib_data['image_size']
        fov_x = 2 * np.arctan(w / (2 * fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(h / (2 * fy)) * 180 / np.pi
        
        print(f"\nEstimated Field of View:")
        print(f"  Horizontal: {fov_x:.1f}°")
        print(f"  Vertical: {fov_y:.1f}°")
        
    except Exception as e:
        print(f"✗ Could not load calibration: {e}")

def main():
    """Main demonstration"""
    print("Fisheye Calibration - Usage Examples")
    print("=" * 50)
    
    # Print calibration summary
    print_calibration_summary()
    
    # Show real-time setup
    example_real_time_undistortion()
    
    # Show batch processing
    example_batch_processing()
    
    print("\n" + "=" * 50)
    print("For more examples, see:")
    print("- README.md: Complete documentation")
    print("- QUICK_START.md: Fast setup guide")
    print("- calibration_utils.py: Validation tools")

if __name__ == "__main__":
    main()