#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <sched.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "yolo_rknn.h"

// --- 配置参数 (参考 Python) ---
#define MODEL_PATH "./Models/cf-11n-rk3588-int8.rknn"
#define HID_DEVICE "/dev/hidg1"
#define CAMERA_INDEX 20
#define CONF_THRES 0.45f
#define NMS_THRES 0.45f

// 控制参数
float PID_KP = 0.55f;
float PID_KD = 0.3f;
float MOUSE_SENSITIVITY = 0.4f;
float HID_SMOOTH_FACTOR = 0.25f;
float AIM_DEADZONE = 0.0f; 
float SHOOT_THRES = 15.0f;
float AIM_HEIGHT_RATIO = 0.12f; // 瞄准高度偏移 (头部)
float SHOOT_PREDICTION = 12.0f; // 射击预测帧数

// 全局状态
std::atomic<bool> running(true);
int hid_fd = -1;

// HID Buffer & 锁
float hid_buffer_x = -20.0f;
float hid_buffer_y = 0.0f;
std::mutex hid_mutex;
std::atomic<bool> should_shoot(false);

// PID 状态
struct PIDState {
    float last_error_x = 0;
    float last_error_y = 0;
    int tracking_frames = 0;
} pid_state;

// --- 系统函数 ---
void set_affinity() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    // 绑定到大核 (4-7)
    CPU_SET(4, &mask);
    CPU_SET(5, &mask);
    CPU_SET(6, &mask);
    CPU_SET(7, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) < 0) {
        std::cerr << "[System] Affinity Error" << std::endl;
    }
}

void hid_init() {
    hid_fd = open(HID_DEVICE, O_RDWR | O_NONBLOCK);
    if (hid_fd < 0) std::cerr << "[HID] Error opening " << HID_DEVICE << std::endl;
    else std::cout << "[HID] Ready" << std::endl;
}

// --- HID 线程 ---
void hid_worker() {
    uint8_t last_buttons = 0;
    
    while (running) {
        if (hid_fd < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        int dx = 0, dy = 0;
        bool shoot = should_shoot.load();
        
        {
            std::lock_guard<std::mutex> lock(hid_mutex);
            
            // 平滑逻辑 (参考 Python)
            // step = buffer * smooth_factor
            float step_x = hid_buffer_x * HID_SMOOTH_FACTOR;
            float step_y = hid_buffer_y * HID_SMOOTH_FACTOR;
            
            // 最小移动量处理
            if (std::abs(hid_buffer_x) > 0.5f && std::abs(step_x) < 1.0f) 
                step_x = (hid_buffer_x > 0) ? 1.0f : -1.0f;
            else if (std::abs(hid_buffer_x) <= 0.5f) 
                step_x = 0;
                
            if (std::abs(hid_buffer_y) > 0.5f && std::abs(step_y) < 1.0f) 
                step_y = (hid_buffer_y > 0) ? 1.0f : -1.0f;
            else if (std::abs(hid_buffer_y) <= 0.5f) 
                step_y = 0;
            
            // 限制单次最大移动 (防止飞鼠标)
            step_x = std::max(-127.0f, std::min(127.0f, step_x));
            step_y = std::max(-127.0f, std::min(127.0f, step_y));
            
            dx = (int)step_x;
            dy = (int)step_y;
            
            // 从 buffer 中减去已移动的量
            hid_buffer_x -= dx;
            hid_buffer_y -= dy;
        }
        
        uint8_t buttons = shoot ? 1 : 0;
        
        // 修复：必须在状态改变时发送报告，否则松开按键的信号发不出去
        if (dx != 0 || dy != 0 || buttons != last_buttons) {
            uint8_t report[] = {buttons, (uint8_t)dx, (uint8_t)dy, 0};
            if (write(hid_fd, report, 4) < 0) {
                // perror("HID Write");
            }
        }
        
        last_buttons = buttons;
        
        // 3ms 间隔 (约 333Hz)
        std::this_thread::sleep_for(std::chrono::microseconds(3000));
    }
}

// --- 图像预处理 ---
void preprocess(const cv::Mat& img, cv::Mat& out, float& r, int& dw, int& dh) {
    float h = img.rows, w = img.cols;
    r = std::min(640.0f/h, 640.0f/w);
    int nw = round(w*r), nh = round(h*r);
    dw = (640-nw)/2; dh = (640-nh)/2;
    
    static cv::Mat resized;
    if (resized.empty() || resized.rows != nh || resized.cols != nw) {
         resized.create(nh, nw, img.type());
    }
    
    cv::resize(img, resized, cv::Size(nw, nh));
    out.setTo(cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(dw, dh, nw, nh)));
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    signal(SIGINT, [](int){ running = false; });
    
    set_affinity();
    hid_init();
    
    // 启动 HID 线程
    std::thread hid_thread(hid_worker);
    
    void* ctx = init_model(MODEL_PATH);
    if (!ctx) {
        std::cerr << "Model Init Failed" << std::endl;
        running = false;
        if(hid_thread.joinable()) hid_thread.join();
        return -1;
    }

    // GStreamer 管道
    std::string pipe = "v4l2src device=/dev/video" + std::to_string(CAMERA_INDEX) + 
                       " ! video/x-raw,width=1920,height=1080,framerate=240/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false max-buffers=1";
    cv::VideoCapture cap(pipe, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cout << "GStreamer failed, trying V4L2..." << std::endl;
        cap.open(CAMERA_INDEX, cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('B', 'G', 'R', '3'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
        cap.set(cv::CAP_PROP_FPS, 180);
    }
    
    int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    float cx = w / 2.0f;
    float cy = h / 2.0f;
    
    std::cout << "[Camera] " << w << "x" << h << std::endl;
    
    cv::Mat frame;
    cv::Mat letterbox_img(640, 640, CV_8UC3);
    cv::Mat rgb_img(640, 640, CV_8UC3);
    Detection dets[100];

    while (running) {
        if (!cap.read(frame)) {
            std::cerr << "Frame drop" << std::endl;
            continue;
        }
        
        float r; int dw, dh;
        preprocess(frame, letterbox_img, r, dw, dh);
        cv::cvtColor(letterbox_img, rgb_img, cv::COLOR_BGR2RGB);
        
        int count = detect(ctx, rgb_img.data, CONF_THRES, NMS_THRES, dets, 100);

        float min_dist = 1e9f; // 修复：初始距离必须足够大，否则边缘目标会被忽略
        Detection* target = nullptr;
        float target_cx = 0, target_cy = 0;
        
        // 寻找最近目标
        for(int i=0; i<count; i++) {
            // 假设 class_id 0 是目标 (根据 Python 代码 TARGET_CLASS_ID = 0)
            if (dets[i].class_id != 0) continue;
            
            // 还原坐标
            float x1 = (dets[i].x1 - dw)/r;
            float y1 = (dets[i].y1 - dh)/r;
            float x2 = (dets[i].x2 - dw)/r;
            float y2 = (dets[i].y2 - dh)/r;
            
            // 计算中心点 (带头部偏移)
            float tcx = (x1 + x2) / 2.0f;
            float tcy = y1 + (y2 - y1) * AIM_HEIGHT_RATIO; // 锁头
            
            float dist = (tcx - cx)*(tcx - cx) + (tcy - cy)*(tcy - cy);
            if (dist < min_dist) { 
                min_dist = dist; 
                target = &dets[i]; 
                target_cx = tcx;
                target_cy = tcy;
            }
        }

        if (target) {
            float raw_dx = target_cx - cx;
            float raw_dy = target_cy - cy;
            
            // PID 计算
            float cd_x = 0, cd_y = 0;
            if (pid_state.tracking_frames > 0) {
                cd_x = raw_dx - pid_state.last_error_x;
                cd_y = raw_dy - pid_state.last_error_y;
            }
            pid_state.tracking_frames++;
            pid_state.last_error_x = raw_dx;
            pid_state.last_error_y = raw_dy;
            
            // 死区
            if (std::abs(raw_dx) < AIM_DEADZONE) raw_dx = 0;
            if (std::abs(raw_dy) < AIM_DEADZONE) raw_dy = 0;
            
            // PD 控制输出
            float move_x = (PID_KP * raw_dx) + (PID_KD * cd_x);
            float move_y = (PID_KP * raw_dy) + (PID_KD * cd_y);
            
            // 累加到 HID Buffer
            {
                std::lock_guard<std::mutex> lock(hid_mutex);
                hid_buffer_x += move_x * MOUSE_SENSITIVITY;
                hid_buffer_y += move_y * MOUSE_SENSITIVITY;
            }
            
            // 自动射击判断 (引入预测逻辑)
            float pred_dx = raw_dx + (cd_x * SHOOT_PREDICTION);
            float pred_dy = raw_dy + (cd_y * SHOOT_PREDICTION);
            
            // 使用矩形范围判断，包含当前位置 OR 预测位置
            if ((std::abs(raw_dx) < SHOOT_THRES && std::abs(raw_dy) < SHOOT_THRES) || 
                (std::abs(pred_dx) < SHOOT_THRES && std::abs(pred_dy) < SHOOT_THRES)) {
                should_shoot = true;
            } else {
                should_shoot = false;
            }
            
        } else {
            // 丢失目标，重置 PID
            pid_state.tracking_frames = 0;
            pid_state.last_error_x = 0;
            pid_state.last_error_y = 0;
            should_shoot = false;
            
            // 可选：丢失目标时是否清空 buffer？Python 代码是清空的
            {
                std::lock_guard<std::mutex> lock(hid_mutex);
                hid_buffer_x = 0;
                hid_buffer_y = 0;
            }
        }
    }
    
    running = false;
    if(hid_thread.joinable()) hid_thread.join();
    
    release_model(ctx);
    if(hid_fd >= 0) close(hid_fd);
    return 0;
}
