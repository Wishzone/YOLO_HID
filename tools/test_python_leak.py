import cv2
import os
import time

def count_fds():
    return len(os.listdir(f'/proc/{os.getpid()}/fd'))

# Use the exact pipeline user claimed works
pipeline = "v4l2src device=/dev/video20 ! video/x-raw,width=1920,height=1080 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false max-buffers=1"

print(f"Opening pipeline via Python: {pipeline}")
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    # Try V4L2 fallback just in case, matching C++ logic
    print("GStreamer failed, trying V4L2...")
    cap.open(20, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('B', 'G', 'R', '3'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

print("Camera opened.")
start_fds = count_fds()

for i in range(1000):
    ret, frame = cap.read()
    if not ret:
        print("Read failed")
        break
    
    if i % 60 == 0:
        print(f"Frame {i}: FDs={count_fds()}")

cap.release()
print("Done")
