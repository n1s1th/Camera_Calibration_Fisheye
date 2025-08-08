from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()
#picam2.configure(picam2.preview_configuration())
picam2.configure(picam2.preview_configuration)


picam2.start()
time.sleep(2)

i = 0
while True:
    frame = picam2.capture_array()
    cv2.imshow("Pi Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        filename = f"calib_images/image_{i}.jpg"
        cv2.imwrite(filename, frame)
        print(f"? Saved {filename}")
        i += 1
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
