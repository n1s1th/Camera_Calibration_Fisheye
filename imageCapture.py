#!/usr/bin/env python3
"""
Enhanced Image Capture for Fisheye Camera Calibration
This script captures calibration images with chessboard detection preview
"""

from picamera2 import Picamera2
import cv2
import time
import os

# Create calibration images directory if it doesn't exist
os.makedirs("calib_images", exist_ok=True)

# Chessboard parameters (adjust based on your calibration pattern)
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners per chessboard row and column

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.preview_configuration())
picam2.start()
time.sleep(2)

# Initialize variables
image_count = 0
print("=== Fisheye Camera Calibration Image Capture ===")
print("Instructions:")
print("1. Hold a chessboard pattern in front of the camera")
print("2. Move it to different positions and angles")
print("3. Press 's' to save image when chessboard is detected (green corners)")
print("4. Capture at least 15-20 images from different angles")
print("5. Press 'q' to quit")
print("6. Ensure the chessboard covers different parts of the fisheye field of view")
print()

try:
    while True:
        frame = picam2.capture_array()
        display_frame = frame.copy()
        
        # Convert to grayscale for chessboard detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        # Draw corners if found
        if ret:
            cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, ret)
            # Add text indicator
            cv2.putText(display_frame, "Chessboard detected - Press 's' to save", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No chessboard detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add image count
        cv2.putText(display_frame, f"Images captured: {image_count}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add instructions
        cv2.putText(display_frame, "Press 's' to save, 'q' to quit", 
                   (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Fisheye Camera Calibration", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            if ret:  # Only save if chessboard is detected
                filename = f"calib_images/calib_image_{image_count:03d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✓ Saved {filename} (chessboard detected)")
                image_count += 1
            else:
                print("✗ Cannot save - no chessboard detected in current frame")
        elif key == ord("q"):
            break

except KeyboardInterrupt:
    print("\nCapture interrupted by user")

finally:
    cv2.destroyAllWindows()
    picam2.stop()
    
    print(f"\nCapture completed. {image_count} images saved.")
    if image_count >= 10:
        print("✓ Good! You have enough images for calibration.")
        print("Run 'python3 fisheye_calibration.py' to start calibration.")
    else:
        print("⚠ Warning: You should capture at least 10-15 images for good calibration results.")
        print("Consider capturing more images from different angles.")
