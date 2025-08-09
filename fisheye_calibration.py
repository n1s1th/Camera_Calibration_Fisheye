#!/usr/bin/env python3
"""
Fisheye Camera Calibration Script
This script performs fisheye camera calibration using OpenCV
Specifically designed for wide-angle 130-degree fisheye cameras
"""

import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path

class FisheyeCalibrator:
    def __init__(self, chessboard_size=(9, 6), square_size=25.0):
        """
        Initialize fisheye calibrator
        
        Args:
            chessboard_size: Tuple of (width, height) of chessboard corners
            square_size: Size of each square in mm
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
    def find_chessboard_corners(self, image_path):
        """Find chessboard corners in an image"""
        img = cv2.imread(image_path)
        if img is None:
            return None, None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        if ret:
            # Refine corner positions for fisheye
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            return True, corners2
        return False, None
    
    def calibrate_fisheye(self, images_path="calib_images/*.jpg"):
        """
        Perform fisheye camera calibration
        
        Args:
            images_path: Path pattern for calibration images
            
        Returns:
            Dictionary containing calibration results
        """
        # Arrays to store object points and image points from all images
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane
        
        images = glob.glob(images_path)
        print(f"Found {len(images)} calibration images")
        
        valid_images = 0
        img_shape = None
        
        for fname in images:
            print(f"Processing {fname}...")
            ret, corners = self.find_chessboard_corners(fname)
            
            if ret:
                objpoints.append(self.objp)
                imgpoints.append(corners)
                valid_images += 1
                
                # Get image dimensions
                if img_shape is None:
                    img = cv2.imread(fname)
                    img_shape = img.shape[:2][::-1]  # (width, height)
                
                print(f"  ✓ Found chessboard corners")
            else:
                print(f"  ✗ Could not find chessboard corners")
        
        if valid_images < 5:
            raise ValueError(f"Need at least 5 valid images, got {valid_images}")
        
        print(f"\nCalibrating fisheye camera with {valid_images} images...")
        
        # Calibrate fisheye camera
        calibration_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + 
                           cv2.fisheye.CALIB_CHECK_COND + 
                           cv2.fisheye.CALIB_FIX_SKEW)
        
        # Initialize camera matrix and distortion coefficients
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]
        
        # Perform calibration
        ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints, imgpoints, img_shape, K, D, rvecs, tvecs,
            calibration_flags, self.criteria)
        
        print(f"Calibration completed with RMS error: {ret:.4f}")
        
        # Prepare results
        calibration_results = {
            'rms_error': ret,
            'camera_matrix': K.tolist(),
            'distortion_coefficients': D.tolist(),
            'image_size': img_shape,
            'valid_images': valid_images,
            'total_images': len(images),
            'chessboard_size': self.chessboard_size,
            'square_size': self.square_size
        }
        
        return calibration_results
    
    def save_calibration(self, calibration_results, filename="calibration_results/fisheye_calibration.json"):
        """Save calibration results to JSON file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(calibration_results, f, indent=4)
        print(f"Calibration results saved to: {filename}")
    
    def undistort_image(self, image_path, calibration_file="calibration_results/fisheye_calibration.json", 
                       output_path=None):
        """
        Undistort an image using calibration results
        
        Args:
            image_path: Path to input image
            calibration_file: Path to calibration JSON file
            output_path: Path for output image (optional)
        """
        # Load calibration data
        with open(calibration_file, 'r') as f:
            calib_data = json.load(f)
        
        K = np.array(calib_data['camera_matrix'])
        D = np.array(calib_data['distortion_coefficients'])
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        h, w = img.shape[:2]
        
        # Generate new camera matrix for undistortion
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=0.0)
        
        # Create undistortion maps
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
        
        # Apply undistortion
        undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_CONSTANT)
        
        # Save result
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"calibration_results/{base_name}_undistorted.jpg"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, undistorted)
        print(f"Undistorted image saved to: {output_path}")
        
        return undistorted

def main():
    """Main function to run fisheye calibration"""
    print("=== Fisheye Camera Calibration ===")
    print("This script will calibrate your 130-degree fisheye camera")
    print()
    
    # Initialize calibrator
    # Standard chessboard: 9x6 corners, 25mm squares
    # Adjust these values based on your calibration pattern
    calibrator = FisheyeCalibrator(chessboard_size=(9, 6), square_size=25.0)
    
    # Check if calibration images exist
    calib_images = glob.glob("calib_images/*.jpg") + glob.glob("calib_images/*.png")
    if len(calib_images) == 0:
        print("Error: No calibration images found in 'calib_images/' directory")
        print("Please capture calibration images first using imageCapture.py")
        return
    
    try:
        # Perform calibration
        results = calibrator.calibrate_fisheye()
        
        # Save results
        calibrator.save_calibration(results)
        
        # Print results summary
        print("\n=== Calibration Results ===")
        print(f"RMS Error: {results['rms_error']:.4f} pixels")
        print(f"Valid Images: {results['valid_images']}/{results['total_images']}")
        print(f"Image Size: {results['image_size']}")
        print()
        print("Camera Matrix (K):")
        K = np.array(results['camera_matrix'])
        print(K)
        print()
        print("Distortion Coefficients (D):")
        D = np.array(results['distortion_coefficients'])
        print(D.flatten())
        
        # Test undistortion on first image
        if calib_images:
            print(f"\nTesting undistortion on: {calib_images[0]}")
            calibrator.undistort_image(calib_images[0])
            
    except Exception as e:
        print(f"Error during calibration: {str(e)}")

if __name__ == "__main__":
    main()