#!/usr/bin/env python3
"""
Calibration Pattern Generator and Validator
This script generates chessboard patterns and validates calibration results
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import os

def generate_chessboard_pattern(size=(9, 6), square_size_mm=25, dpi=300, output_file="sample_patterns/chessboard_9x6.png"):
    """
    Generate a printable chessboard pattern
    
    Args:
        size: (width, height) number of inner corners
        square_size_mm: Size of each square in millimeters
        dpi: Print resolution
        output_file: Output file path
    """
    # Calculate dimensions
    squares_per_inch = 25.4 / square_size_mm  # squares per inch
    pixels_per_square = int(dpi / squares_per_inch)
    
    # Total dimensions
    pattern_width = (size[0] + 1) * pixels_per_square
    pattern_height = (size[1] + 1) * pixels_per_square
    
    # Create pattern
    pattern = np.zeros((pattern_height, pattern_width), dtype=np.uint8)
    
    for i in range(size[1] + 1):
        for j in range(size[0] + 1):
            if (i + j) % 2 == 0:
                y_start = i * pixels_per_square
                y_end = (i + 1) * pixels_per_square
                x_start = j * pixels_per_square
                x_end = (j + 1) * pixels_per_square
                pattern[y_start:y_end, x_start:x_end] = 255
    
    # Save pattern
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cv2.imwrite(output_file, pattern)
    
    print(f"Generated chessboard pattern: {output_file}")
    print(f"Pattern size: {size[0]}x{size[1]} corners")
    print(f"Square size: {square_size_mm}mm")
    print(f"Print dimensions: {pattern_width/dpi:.2f}\" x {pattern_height/dpi:.2f}\" inches")
    print(f"Resolution: {dpi} DPI")
    
    return output_file

def validate_calibration(calibration_file="calibration_results/fisheye_calibration.json"):
    """
    Validate and analyze calibration results
    
    Args:
        calibration_file: Path to calibration JSON file
    """
    if not os.path.exists(calibration_file):
        print(f"Calibration file not found: {calibration_file}")
        return
    
    # Load calibration data
    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)
    
    K = np.array(calib_data['camera_matrix'])
    D = np.array(calib_data['distortion_coefficients'])
    
    print("=== Calibration Validation Report ===")
    print(f"RMS Error: {calib_data['rms_error']:.4f} pixels")
    
    # Quality assessment
    if calib_data['rms_error'] < 0.5:
        quality = "Excellent"
    elif calib_data['rms_error'] < 1.0:
        quality = "Good"
    elif calib_data['rms_error'] < 2.0:
        quality = "Acceptable"
    else:
        quality = "Poor - Consider Recalibration"
    
    print(f"Calibration Quality: {quality}")
    print(f"Valid Images Used: {calib_data['valid_images']}/{calib_data['total_images']}")
    print(f"Image Resolution: {calib_data['image_size']}")
    
    print(f"\nCamera Parameters:")
    print(f"Focal Length X: {K[0,0]:.2f} pixels")
    print(f"Focal Length Y: {K[1,1]:.2f} pixels")
    print(f"Principal Point: ({K[0,2]:.2f}, {K[1,2]:.2f})")
    
    print(f"\nDistortion Coefficients:")
    for i, coeff in enumerate(D.flatten()):
        print(f"k{i+1}: {coeff:.6f}")
    
    # Field of view estimation
    w, h = calib_data['image_size']
    fov_x = 2 * np.arctan(w / (2 * K[0,0])) * 180 / np.pi
    fov_y = 2 * np.arctan(h / (2 * K[1,1])) * 180 / np.pi
    
    print(f"\nEstimated Field of View:")
    print(f"Horizontal FOV: {fov_x:.1f}°")
    print(f"Vertical FOV: {fov_y:.1f}°")
    print(f"Diagonal FOV: ~{np.sqrt(fov_x**2 + fov_y**2):.1f}°")

def create_undistortion_comparison(image_path, calibration_file="calibration_results/fisheye_calibration.json", 
                                 output_file="calibration_results/comparison.jpg"):
    """
    Create side-by-side comparison of original and undistorted image
    
    Args:
        image_path: Input image path
        calibration_file: Calibration data file
        output_file: Output comparison image
    """
    if not os.path.exists(calibration_file):
        print(f"Calibration file not found: {calibration_file}")
        return
    
    # Load calibration data
    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)
    
    K = np.array(calib_data['camera_matrix'])
    D = np.array(calib_data['distortion_coefficients'])
    
    # Load and undistort image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    h, w = img.shape[:2]
    
    # Generate undistortion maps
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.0)
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    
    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_CONSTANT)
    
    # Create side-by-side comparison
    comparison = np.hstack((img, undistorted))
    
    # Add labels
    cv2.putText(comparison, "Original (Distorted)", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(comparison, "Undistorted", (w + 20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    # Save comparison
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cv2.imwrite(output_file, comparison)
    print(f"Comparison image saved: {output_file}")

def main():
    """Main function with utility options"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 calibration_utils.py [generate|validate|compare]")
        print("\nCommands:")
        print("  generate - Generate printable chessboard pattern")
        print("  validate - Validate calibration results")
        print("  compare  - Create before/after undistortion comparison")
        return
    
    command = sys.argv[1]
    
    if command == "generate":
        print("Generating chessboard calibration pattern...")
        generate_chessboard_pattern()
        print("\nPrint the generated pattern and use it for calibration.")
        
    elif command == "validate":
        print("Validating calibration results...")
        validate_calibration()
        
    elif command == "compare":
        if len(sys.argv) < 3:
            print("Usage: python3 calibration_utils.py compare <image_path>")
            return
        image_path = sys.argv[2]
        print(f"Creating undistortion comparison for: {image_path}")
        create_undistortion_comparison(image_path)
        
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()