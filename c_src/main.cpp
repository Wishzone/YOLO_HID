#include <iostream>
#include <thread>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "yolo_rknn.h"

#define MODEL_PATH "./Models/cf-11n-rk3588-int8.rknn"
#define HID_DEVICE "/dev/hidg1"
#define CAMERA_INDEX 20
#define CONF_THRES 0.45f
#define NMS_THRES 0.45f
#define SENSITIVITY 0.4f
#define SMOOTH 0.3f
#define SHOOT_THRES 20.0f

std::atomic<bool> running(true);
int hid_fd = -1;
float last_dx = 0, last_dy = 0;

void hid_init() {
    hid_fd = open(HID_DEVICE, O_RDWR | O_NONBLOCK);
    if (hid_fd < 0) std::cerr << "HID Error" << std::endl;
}

void hid_move(float dx, float dy, bool click) {
    if (hid_fd < 0) return;
    float mx = dx * SENSITIVITY + (dx - last_dx) * SMOOTH;
    float my = dy * SENSITIVITY + (dy - last_dy) * SMOOTH;
    last_dx = dx; last_dy = dy;
    int8_t ix = std::max(-127.0f, std::min(127.0f, mx));
    int8_t iy = std::max(-127.0f, std::min(127.0f, my));
    uint8_t report[] = {(uint8_t)(click ? 1 : 0), (uint8_t)ix, (uint8_t)iy, 0};
    if(write(hid_fd, report, 4) < 0) {};
}

cv::Mat preprocess(const cv::Mat& img, float& r, int& dw, int& dh) {
    float h = img.rows, w = img.cols;
    r = std::min(640.0f/h, 640.0f/w);
    int nw = round(w*r), nh = round(h*r);
    dw = (640-nw)/2; dh = (640-nh)/2;
    cv::Mat resized, out;
    cv::resize(img, resized, cv::Size(nw, nh));
    cv::copyMakeBorder(resized, out, dh, dh, dw, dw, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
    return out;
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    signal(SIGINT, [](int){ running = false; });
    
    void* ctx = init_model(MODEL_PATH);
    if (!ctx) return -1;
    
    hid_init();

    std::string pipe = "v4l2src device=/dev/video" + std::to_string(CAMERA_INDEX) + " ! video/x-raw,width=1920,height=1080 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false max-buffers=1";
    cv::VideoCapture cap(pipe, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        cap.open(CAMERA_INDEX, cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('B', 'G', 'R', '3'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
        cap.set(cv::CAP_PROP_FPS, 60);
    }
    
    int w = cap.get(cv::CAP_PROP_FRAME_WIDTH), h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    float cx = w/2.0f, cy = h/2.0f;
    cv::Mat frame, rgb;
    Detection dets[100];

    while (running && cap.read(frame)) {
        float r; int dw, dh;
        cv::cvtColor(preprocess(frame, r, dw, dh), rgb, cv::COLOR_BGR2RGB);
        int count = detect(ctx, rgb.data, CONF_THRES, NMS_THRES, dets, 100);

        float min_dist = 10000;
        Detection* target = nullptr;
        
        for(int i=0; i<count; i++) {
            if (dets[i].class_id != 0) continue;
            float bx = (dets[i].x1 - dw)/r, by = (dets[i].y1 - dh)/r;
            float bw = (dets[i].x2 - dw)/r - bx, bh = (dets[i].y2 - dh)/r - by;
            float tcx = bx + bw/2, tcy = by + bh/2;
            float dist = hypot(tcx - cx, tcy - cy);
            if (dist < min_dist) { min_dist = dist; target = &dets[i]; }
        }

        if (target) {
            float tcx = (target->x1 - dw)/r + ((target->x2 - target->x1)/r)/2;
            float tcy = (target->y1 - dh)/r + ((target->y2 - target->y1)/r)/2;
            hid_move(tcx - cx, tcy - cy, min_dist < SHOOT_THRES);
        }
    }
    
    release_model(ctx);
    if(hid_fd >= 0) close(hid_fd);
    return 0;
}
