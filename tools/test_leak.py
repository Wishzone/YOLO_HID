import cv2
import os
import time

def count_fds():
    try:
        return len(os.listdir(f'/proc/{os.getpid()}/fd'))
    except:
        return 0

# Pipeline matching the C++ one as closely as possible
pipeline = "v4l2src device=/dev/video20 io-mode=2 ! video/x-raw,width=1920,height=1080,framerate=60/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false"

print(f"Opening pipeline: {pipeline}")
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open camera via GStreamer")
    exit(1)

print("Camera opened. persistent loop...")
start_fds = count_fds()
print(f"Start FDs: {start_fds}")

for i in range(300):
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed")
        break
    
    if i % 50 == 0:
        curr_fds = count_fds()
        print(f"Frame {i}: FDs={curr_fds}")

cap.release()
print("Done")
