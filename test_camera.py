import cv2
import time

def test_camera():
    print("Testing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    print("Camera opened successfully")
    
    # Try to read 10 frames
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i+1}: shape={frame.shape}, dtype={frame.dtype}")
            
            # Save the first frame as a test
            if i == 0:
                cv2.imwrite('test_frame.jpg', frame)
                print("Saved test frame as 'test_frame.jpg'")
        else:
            print(f"Failed to read frame {i+1}")
        time.sleep(0.1)
    
    cap.release()
    print("Test complete")

if __name__ == "__main__":
    test_camera() 