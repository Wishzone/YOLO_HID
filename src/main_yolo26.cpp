#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <signal.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <rga/im2d.h>
#include "models/yolo26_rknn.h"
#include "modules/v4l2_driver.h"
#include "core/globals.h"
#include "utils/sys_utils.h"
#include "modules/hid_mouse.h"
#include "modules/web_server.h"
#include "modules/camera_capture.h"

// --- Configuration ---
#define MODEL_PATH "./Models/cf-26n-rk3588-int8_100.rknn"
#define HID_DEVICE "/dev/hidg1"
#define CAMERA_INDEX 20
#define CONF_THRES 0.70f
#define HTTP_PORT 8080
#define MJPEG_QUALITY 80 

// Control Parameters
const float PID_KP = 0.60f;
const float PID_KD = 0.35f;
const float MOUSE_SENSITIVITY = 0.4f;
const float HID_SMOOTH_FACTOR = 1.0f;
const float AIM_DEADZONE = 0.0f; 
const float AIM_HEIGHT_RATIO = 0.20f;
const float AUTO_FIRE_RADIUS = 15.0f; 
const int MAX_DETECTIONS = 10;

// PID State
struct {
    float last_error_x = 0;
    float last_error_y = 0;
    int tracking_frames = 0;
} pid_state;

// --- Main Loop ---
int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    signal(SIGINT, [](int){ running = false; hid_cv.notify_all(); cap_state.cv.notify_all(); });
    signal(SIGPIPE, SIG_IGN);
    
    set_affinity_big(); 
    set_realtime_priority();
    lock_memory();
    hid_init(HID_DEVICE);
    
    std::thread hid_thread([](){ set_affinity_little(); hid_worker(HID_SMOOTH_FACTOR); });
    std::thread web_thread([](){ web_worker(HTTP_PORT, MJPEG_QUALITY, AIM_HEIGHT_RATIO); });
    std::thread cleaner_thread([](){ // Background FD Scrubber
        set_affinity_little();
        while(running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            scrub_sync_files();
        }
    });
    cleaner_thread.detach(); 
    
    void* ctx = init_yolo26_model(MODEL_PATH);
    if (!ctx) { std::cerr << "Model Init Failed" << std::endl; return -1; }

    v4l2_context_t camera_ctx;
    // Open Camera (Request 180hz)
    if (v4l2_open(&camera_ctx, "/dev/video20", 1920, 1080, 180) < 0) return -1;
    v4l2_start_stream(&camera_ctx);
    std::cout << "[System] Manual V4L2 Driver Started" << std::endl;
    
    std::thread cap_thread(capture_worker, &camera_ctx);
    std::thread web_proc_thread(web_processor_func, &camera_ctx); // Launch processor
    
    cv::Mat letterbox_img(640, 640, CV_8UC3);
    cv::Mat rgb_img(640, 640, CV_8UC3);
    rgb_img.setTo(cv::Scalar(114, 114, 114));
    
    Detection dets[50];
    auto last_print_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    
    const float cx = 1920 / 2.0f;
    const float cy = 1080 / 2.0f;

    // 预计算参数
    float r = std::min(640.0f/1080.0f, 640.0f/1920.0f);
    int nw = round(1920*r), nh = round(1080*r);
    int dw = (640-nw)/2, dh = (640-nh)/2;

    while (running) {
        auto t_start = std::chrono::high_resolution_clock::now();
        
        int current_idx = -1;
        {
            std::unique_lock<std::mutex> lock(cap_state.mutex);
            if (!cap_state.new_frame_available) cap_state.cv.wait(lock);
            if (cap_state.latest_idx != -1) {
                current_idx = cap_state.latest_idx;
                cap_state.latest_idx = -1; 
                cap_state.new_frame_available = false;
            }
        }
        
        if (current_idx == -1) continue; 
        
        // Zero-Copy RGA Preprocess
        int dma_fd = camera_ctx.buffers[current_idx].dma_fd;
        void* raw_data = camera_ctx.buffers[current_idx].start;
        
        if (camera_ctx.format == V4L2_PIX_FMT_BGR24) {
            rga_buffer_t src_rga;
            if (dma_fd >= 0) src_rga = wrapbuffer_fd(dma_fd, 1920, 1080, RK_FORMAT_BGR_888);
            else src_rga = wrapbuffer_virtualaddr(raw_data, 1920, 1080, RK_FORMAT_BGR_888);
            
            void* dst_ptr = rgb_img.data + (dh * 640 + dw) * 3;
            rga_buffer_t dst_rga = wrapbuffer_virtualaddr(dst_ptr, nw, nh, RK_FORMAT_RGB_888); 
            dst_rga.wstride = 640; dst_rga.hstride = nh;
            
            if (imresize(src_rga, dst_rga) != IM_STATUS_SUCCESS) {
                cv::Mat wrapper(1080, 1920, CV_8UC3, raw_data);
                preprocess(wrapper, letterbox_img, r, dw, dh);
                cv::cvtColor(letterbox_img, rgb_img, cv::COLOR_BGR2RGB);
            }
        } else {
             cv::Mat wrapper(1080, 1920, CV_8UC3, raw_data); // Assumption: BGR24 or MJPG (but wrapped as raw)
             preprocess(wrapper, letterbox_img, r, dw, dh);
             cv::cvtColor(letterbox_img, rgb_img, cv::COLOR_BGR2RGB);
        }

        // Inference
        int count = detect_yolo26(ctx, rgb_img.data, CONF_THRES, dets, MAX_DETECTIONS);

        // Tracker & PID
        float min_dist = 1e9f;
        Detection* target = nullptr;
        float target_cx = 0, target_cy = 0;
        
        for(int i=0; i<count; i++) {
            if (dets[i].class_id != 0) continue;

            float x1 = (dets[i].x1 - dw)/r, y1 = (dets[i].y1 - dh)/r;
            float x2 = (dets[i].x2 - dw)/r, y2 = (dets[i].y2 - dh)/r;
            float tcx = (x1 + x2) / 2.0f;
            float tcy = y1 + (y2 - y1) * AIM_HEIGHT_RATIO;
            float dist = (tcx - cx)*(tcx - cx) + (tcy - cy)*(tcy - cy);
            if (dist < min_dist) { min_dist = dist; target = &dets[i]; target_cx = tcx; target_cy = tcy; }
        }

        if (target) {
            float raw_dx = target_cx - cx;
            float raw_dy = target_cy - cy;
            
            if (std::abs(raw_dx) < AIM_DEADZONE) raw_dx = 0;
            if (std::abs(raw_dy) < AIM_DEADZONE) raw_dy = 0;

            float mx = (PID_KP * raw_dx) + (PID_KD * (raw_dx - pid_state.last_error_x));
            float my = (PID_KP * raw_dy) + (PID_KD * (raw_dy - pid_state.last_error_y));
            
            pid_state.last_error_x = raw_dx; pid_state.last_error_y = raw_dy;
            pid_state.tracking_frames++;
            
            float predicted_dist = std::sqrt(std::pow(raw_dx * (1.0f - PID_KP), 2) + std::pow(raw_dy * (1.0f - PID_KP), 2));
            bool safe_to_predict = std::sqrt(min_dist) < (AUTO_FIRE_RADIUS * 2.5f);
            bool should_fire = (std::sqrt(min_dist) < AUTO_FIRE_RADIUS) || (predicted_dist < AUTO_FIRE_RADIUS && safe_to_predict);
            
            if (fire_request != should_fire) {
                fire_request = should_fire;
                hid_cv.notify_one();
            }

            {
                std::lock_guard<std::mutex> lock(hid_mutex);
                hid_buffer_x += mx * MOUSE_SENSITIVITY;
                hid_buffer_y += my * MOUSE_SENSITIVITY;
            }
            hid_cv.notify_one();
        } else {
            pid_state.tracking_frames = 0;
            pid_state.last_error_x = 0;
            pid_state.last_error_y = 0;
            if (fire_request) {
                 fire_request = false;
                 hid_cv.notify_one();
            }
            std::lock_guard<std::mutex> lock(hid_mutex);
            hid_buffer_x = 0; hid_buffer_y = 0;
        }

        // 5. Offload Buffer to Web Thread (If available)
        bool buffer_offloaded = false;
        
        if (active_connections > 0) {
             int expected = -1;
             // Only try if web thread is asking/idle (idx == -1)
             if (std::atomic_load(&web_transfer_buf_idx) == -1) {
                // Populate params safely before swap
                {
                    std::lock_guard<std::mutex> lock(web_transfer_mutex);
                    web_transfer_dets.assign(dets, dets + count);
                }
                web_transfer_r = r;
                web_transfer_dw = dw;
                web_transfer_dh = dh;
                web_transfer_src_dma = dma_fd;
                web_transfer_src_w = camera_ctx.width;
                web_transfer_src_h = camera_ctx.height;
                web_transfer_length = 0; // Not used for BGR24 path usually
                web_transfer_data = raw_data;
                
                if (web_transfer_buf_idx.compare_exchange_strong(expected, current_idx)) {
                    buffer_offloaded = true;
                }
             }
        }
        
        if (!buffer_offloaded) {
             v4l2_manual_qbuf(&camera_ctx, current_idx);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        frame_count++;

        if (std::chrono::duration<double>(t_end - last_print_time).count() >= 1.0) {
            std::cout << std::fixed << std::setprecision(1) << "[FPS: " << frame_count << "]\r" << std::flush;
            last_print_time = t_end; frame_count=0;
        }
    }
    
    running = false;
    if(hid_thread.joinable()) hid_thread.join();
    if(web_thread.joinable()) web_thread.join();
    if(web_proc_thread.joinable()) web_proc_thread.join();
    if(cap_thread.joinable()) cap_thread.join();
    
    release_yolo26_model(ctx);
    v4l2_close(&camera_ctx);
    return 0;
}
