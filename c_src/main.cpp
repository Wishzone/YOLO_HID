#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <sys/select.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "yolo_rknn.h"

// --- Configuration ---
#define MODEL_PATH "./Models/cf-11n-rk3588-int8.rknn"
#define HID_DEVICE "/dev/hidg1"
#define CAMERA_INDEX 20
#define CONF_THRES 0.45f
#define NMS_THRES 0.45f
#define TARGET_CLASS_ID 0 // 0 for person, 1 for whatever custom class
#define PORT 5000

// PID & Aiming Config
#define AIM_OFFSET_X 0
#define AIM_OFFSET_Y 0
#define MOUSE_SENSITIVITY 0.4f
#define HID_SMOOTH_FACTOR 0.3f
#define SHOOT_THRESHOLD 20.0f // pixels

// Globals
std::atomic<bool> running(true);
std::mutex display_mutex;
cv::Mat display_frame;

// --- HID Controller ---
class HIDController {
    int fd;
    float last_error_x = 0, last_error_y = 0;
    
public:
    HIDController() {
        fd = open(HID_DEVICE, O_RDWR | O_NONBLOCK);
        if (fd < 0) {
            std::cerr << "[HID] Failed to open " << HID_DEVICE << " (Error: " << errno << ")" << std::endl;
        } else {
            std::cout << "[HID] Ready: " << HID_DEVICE << std::endl;
        }
    }

    ~HIDController() {
        if (fd >= 0) close(fd);
    }

    void move_and_click(float dx, float dy, bool click) {
        if (fd < 0) return;

        // PID / Smoothing
        float move_x = dx * MOUSE_SENSITIVITY + (dx - last_error_x) * HID_SMOOTH_FACTOR;
        float move_y = dy * MOUSE_SENSITIVITY + (dy - last_error_y) * HID_SMOOTH_FACTOR;
        
        last_error_x = dx;
        last_error_y = dy;

        // Clamp to signed 8-bit (-127 to 127)
        int8_t ix = (int8_t)std::max(-127.0f, std::min(127.0f, move_x));
        int8_t iy = (int8_t)std::max(-127.0f, std::min(127.0f, move_y));
        uint8_t buttons = click ? 0x01 : 0x00;

        uint8_t report[4] = {buttons, (uint8_t)ix, (uint8_t)iy, 0};
        write(fd, report, 4);
    }
};

// --- MJPEG Server ---
void mjpeg_server() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        return;
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        return;
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        return;
    }

    if (listen(server_fd, 3) < 0) {
        perror("listen");
        return;
    }

    std::cout << "[Web] Listening on port " << PORT << std::endl;

    while (running) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(server_fd, &readfds);
        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;

        int activity = select(server_fd + 1, &readfds, NULL, NULL, &timeout);

        if (activity < 0 && errno != EINTR) {
            perror("select error");
            break;
        }

        if (activity == 0) {
            continue;
        }

        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
            continue;
        }

        std::string header = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
        send(new_socket, header.c_str(), header.size(), 0);

        while (running) {
            cv::Mat raw;
            {
                std::lock_guard<std::mutex> lock(display_mutex);
                if (display_frame.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    continue;
                }
                raw = display_frame.clone();
            }

            std::vector<uchar> buf;
            // Encode in Web Thread to avoid blocking Main Thread
            cv::imencode(".jpg", raw, buf);

            std::string boundary = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + std::to_string(buf.size()) + "\r\n\r\n";
            if (send(new_socket, boundary.c_str(), boundary.size(), MSG_NOSIGNAL) < 0) break;
            if (send(new_socket, buf.data(), buf.size(), MSG_NOSIGNAL) < 0) break;
            if (send(new_socket, "\r\n", 2, MSG_NOSIGNAL) < 0) break;
        }
        close(new_socket);
    }
    close(server_fd);
}

// --- Helper: Letterbox ---
cv::Mat letterbox(const cv::Mat& img, int new_shape, float& ratio, int& dw, int& dh) {
    int shape_w = img.cols;
    int shape_h = img.rows;
    float r = std::min((float)new_shape / shape_h, (float)new_shape / shape_w);
    
    int new_unpad_w = (int)round(shape_w * r);
    int new_unpad_h = (int)round(shape_h * r);
    
    dw = new_shape - new_unpad_w;
    dh = new_shape - new_unpad_h;
    dw /= 2;
    dh /= 2;

    cv::Mat resized;
    if (shape_w != new_unpad_w || shape_h != new_unpad_h) {
        cv::resize(img, resized, cv::Size(new_unpad_w, new_unpad_h));
    } else {
        resized = img;
    }

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, dh, dh, dw, dw, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    // Ensure 640x640 exactly (handle rounding errors)
    if (padded.rows != new_shape || padded.cols != new_shape) {
        cv::resize(padded, padded, cv::Size(new_shape, new_shape));
    }
    
    ratio = r;
    return padded;
}

void signal_handler(int s) {
    running = false;
}

int main() {
    // Suppress Logs
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    setenv("RKNN_LOG_LEVEL", "0", 1);
    
    signal(SIGINT, signal_handler);

    // 1. Init Model
    void* ctx = init_model(MODEL_PATH);
    if (!ctx) {
        std::cerr << "Failed to init RKNN model" << std::endl;
        return -1;
    }

    // 2. Init Camera
    // Try GStreamer first for low latency
    std::string gst_pipeline = "v4l2src device=/dev/video" + std::to_string(CAMERA_INDEX) + 
                               " ! video/x-raw,width=1920,height=1080 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false max-buffers=1";
    cv::VideoCapture cap(gst_pipeline, cv::CAP_GSTREAMER);
    
    if (!cap.isOpened()) {
        std::cout << "[Camera] GStreamer failed, falling back to V4L2..." << std::endl;
        cap.open(CAMERA_INDEX, cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('B', 'G', 'R', '3'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
        cap.set(cv::CAP_PROP_FPS, 60);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    }

    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
        return -1;
    }

    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "[Camera] " << width << "x" << height << std::endl;

    // 3. Init HID
    HIDController hid;

    // 4. Start Web Server
    std::thread web_thread(mjpeg_server);

    // 5. Main Loop
    cv::Mat frame, input_img;
    Detection dets[100];
    float center_x = width / 2.0f + AIM_OFFSET_X;
    float center_y = height / 2.0f + AIM_OFFSET_Y;

    auto last_time = std::chrono::high_resolution_clock::now();

    while (running) {
        cap >> frame;
        if (frame.empty()) break;

        // Preprocess
        float ratio;
        int dw, dh;
        input_img = letterbox(frame, 640, ratio, dw, dh);
        
        // Convert to RGB for model (if model expects RGB)
        // Assuming model expects RGB based on typical YOLO training
        cv::cvtColor(input_img, input_img, cv::COLOR_BGR2RGB);

        // Inference
        int count = detect(ctx, input_img.data, CONF_THRES, NMS_THRES, dets, 100);

        // Process Detections
        float min_dist = 10000.0f;
        Detection* target = nullptr;

        // 1. Find Target (Fast Pass)
        for (int i = 0; i < count; i++) {
            // Restore coordinates
            dets[i].x1 = (dets[i].x1 - dw) / ratio;
            dets[i].y1 = (dets[i].y1 - dh) / ratio;
            dets[i].x2 = (dets[i].x2 - dw) / ratio;
            dets[i].y2 = (dets[i].y2 - dh) / ratio;

            // Find Target
            if (dets[i].class_id == TARGET_CLASS_ID) {
                float cx = (dets[i].x1 + dets[i].x2) / 2.0f;
                float cy = (dets[i].y1 + dets[i].y2) / 2.0f;
                float dist = std::sqrt(std::pow(cx - center_x, 2) + std::pow(cy - center_y, 2));
                
                if (dist < min_dist) {
                    min_dist = dist;
                    target = &dets[i];
                }
            }
        }

        // 2. Aim & Shoot (Immediate Action)
        if (target) {
            float cx = (target->x1 + target->x2) / 2.0f;
            float cy = (target->y1 + target->y2) / 2.0f;
            float dx = cx - center_x;
            float dy = cy - center_y;
            
            bool shoot = (min_dist < SHOOT_THRESHOLD);
            hid.move_and_click(dx, dy, shoot);
        }

        // 3. Draw Debug Info (Delayed)
        for (int i = 0; i < count; i++) {
             cv::rectangle(frame, cv::Point(dets[i].x1, dets[i].y1), cv::Point(dets[i].x2, dets[i].y2), cv::Scalar(0, 255, 0), 2);
        }
        if (target) {
            float cx = (target->x1 + target->x2) / 2.0f;
            float cy = (target->y1 + target->y2) / 2.0f;
            cv::line(frame, cv::Point(center_x, center_y), cv::Point(cx, cy), cv::Scalar(0, 0, 255), 2);
        }

        // FPS Calculation
        auto now = std::chrono::high_resolution_clock::now();
        float fps = 1000.0f / std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
        last_time = now;
        cv::putText(frame, "FPS: " + std::to_string((int)fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        // Update Stream (Non-blocking copy)
        {
            std::lock_guard<std::mutex> lock(display_mutex);
            display_frame = frame.clone();
        }
    }

    release_model(ctx);
    running = false;
    web_thread.join();
    return 0;
}
