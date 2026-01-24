#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cmath>
#include <algorithm>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <sched.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <rga/im2d.h>
#include "yolo_rknn.h"
#include "v4l2_driver.h"

// --- Configuration ---
#define MODEL_PATH "./Models/cf-11n-rk3588-int8.rknn"
#define HID_DEVICE "/dev/hidg1"
#define CAMERA_INDEX 20
#define CONF_THRES 0.2f
#define NMS_THRES 0.45f
#define HTTP_PORT 8080
#define MJPEG_QUALITY 80 
#define WEB_MAX_FPS 15

// Control Parameters
const float PID_KP = 0.55f;
const float PID_KD = 0.3f;
const float MOUSE_SENSITIVITY = 0.4f;
const float HID_SMOOTH_FACTOR = 0.3f;
const float AIM_DEADZONE = 0.0f; 
const float AIM_HEIGHT_RATIO = 0.12f;

// --- Global State ---
std::atomic<bool> running(true);
std::atomic<int> active_connections(0);
int hid_fd = -1;

// HID Buffers
float hid_buffer_x = 0.0f;
float hid_buffer_y = 0.0f;
std::mutex hid_mutex;
std::condition_variable hid_cv;

// Web Display Data
struct WebData {
    cv::Mat img;
    std::vector<Detection> dets;
    float r; int dw, dh;
    bool valid = false;
} web_data_buffer;
std::mutex web_mutex;

// PID State
struct {
    float last_error_x = 0;
    float last_error_y = 0;
    int tracking_frames = 0;
} pid_state;

// --- System Utilities ---
void set_realtime_priority() {
    struct sched_param param;
    param.sched_priority = 50; 
    if (sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
        std::cerr << "[System] RT Priority Failed (Run as Root)" << std::endl;
    } else {
        std::cout << "[System] Real-Time Priority Enabled" << std::endl;
    }
}

void lock_memory() {
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        std::cerr << "[System] Memory Lock Failed" << std::endl;
    }
}

void set_affinity_big() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(4, &mask); CPU_SET(5, &mask); CPU_SET(6, &mask); CPU_SET(7, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
}

void set_affinity_little() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask); CPU_SET(1, &mask); CPU_SET(2, &mask); CPU_SET(3, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
}

// --- Workers ---

void hid_init() {
    hid_fd = open(HID_DEVICE, O_RDWR | O_NONBLOCK);
    if (hid_fd < 0) std::cerr << "[HID] Error opening " << HID_DEVICE << std::endl;
}

void hid_worker() {
    struct pollfd pfd;
    pfd.fd = hid_fd;
    pfd.events = POLLOUT;
    
    while (running) {
        if (hid_fd < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(hid_mutex);
            if (std::abs(hid_buffer_x) < 0.5f && std::abs(hid_buffer_y) < 0.5f) {
                hid_cv.wait(lock);
            }
        }
        
        if (!running) break;
        if (poll(&pfd, 1, 5) <= 0) continue; 

        int dx = 0, dy = 0;
        {
            std::lock_guard<std::mutex> lock(hid_mutex);
            float step_x = hid_buffer_x * HID_SMOOTH_FACTOR;
            float step_y = hid_buffer_y * HID_SMOOTH_FACTOR;
            
            // Min movement logic
            if (std::abs(hid_buffer_x) > 0.5f && std::abs(step_x) < 1.0f) 
                step_x = (hid_buffer_x > 0) ? 1.0f : -1.0f;
            else if (std::abs(hid_buffer_x) <= 0.5f) step_x = 0;
                
            if (std::abs(hid_buffer_y) > 0.5f && std::abs(step_y) < 1.0f) 
                step_y = (hid_buffer_y > 0) ? 1.0f : -1.0f;
            else if (std::abs(hid_buffer_y) <= 0.5f) step_y = 0;
            
            step_x = std::max(-127.0f, std::min(127.0f, step_x));
            step_y = std::max(-127.0f, std::min(127.0f, step_y));
            
            dx = (int)step_x;
            dy = (int)step_y;
            
            if (dx != 0 || dy != 0) {
                uint8_t report[] = {0, (uint8_t)dx, (uint8_t)dy, 0}; 
                if (write(hid_fd, report, 4) == 4) {
                    hid_buffer_x -= dx;
                    hid_buffer_y -= dy;
                }
            }
        }
    }
}

// CPU Fallback Preprocess
void preprocess(const cv::Mat& img, cv::Mat& out, float& r, int& dw, int& dh) {
    float h = img.rows, w = img.cols;
    r = std::min(640.0f/h, 640.0f/w);
    int nw = round(w*r), nh = round(h*r);
    dw = (640-nw)/2; dh = (640-nh)/2;
    
    static cv::Mat resized;
    if (resized.empty() || resized.rows != nh || resized.cols != nw) resized.create(nh, nw, img.type());
    
    cv::resize(img, resized, cv::Size(nw, nh));
    out.setTo(cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(dw, dh, nw, nh)));
}

// Async Capture State
struct {
    std::mutex mutex;
    std::condition_variable cv;
    int latest_idx = -1;
    bool new_frame_available = false;
    v4l2_context_t* ctx = nullptr;
} cap_state;

void capture_worker(v4l2_context_t* ctx) {
    cap_state.ctx = ctx;
    while (running) {
        struct v4l2_buffer buf;
        struct v4l2_plane planes[8];
        buf.m.planes = planes; 
        
        fd_set fds;
        struct timeval tv;
        FD_ZERO(&fds);
        FD_SET(ctx->fd, &fds);
        tv.tv_sec = 1; tv.tv_usec = 0;
        
        if (select(ctx->fd + 1, &fds, NULL, NULL, &tv) <= 0) continue;
        
        if (v4l2_manual_dqbuf(ctx, &buf) == 0) {
            {
                std::lock_guard<std::mutex> lock(cap_state.mutex);
                int old_idx = cap_state.latest_idx;
                cap_state.latest_idx = buf.index;
                cap_state.new_frame_available = true;
                if (old_idx != -1) v4l2_manual_qbuf(ctx, old_idx);
            }
            cap_state.cv.notify_one();
        }
    }
}

// Web Server
void handle_client(int client_socket); // Forward decl
void web_worker() {
    set_affinity_little();
    int server_fd;
    struct sockaddr_in address;
    int opt = 1;

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(HTTP_PORT);
    
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) return;
    listen(server_fd, 5);

    while (running) {
        struct pollfd pfd = {server_fd, POLLIN, 0};
        if (poll(&pfd, 1, 500) > 0) {
            int addrlen = sizeof(address);
            int new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
            if (new_socket >= 0) {
                if (active_connections >= 5) close(new_socket);
                else std::thread(handle_client, new_socket).detach();
            }
        }
    }
    close(server_fd);
}

void handle_client(int client_socket) {
    active_connections++;
    struct timeval tv = {3, 0};
    setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(client_socket, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    char buffer[1024] = {0};
    if (read(client_socket, buffer, 1024) <= 0) { close(client_socket); active_connections--; return; }
    
    if (std::string(buffer).find("GET /stream") != std::string::npos) {
        std::string h = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
        send(client_socket, h.c_str(), h.length(), MSG_NOSIGNAL);

        std::vector<uchar> buf;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, MJPEG_QUALITY};
        cv::Mat local_frame;
        std::vector<Detection> local_dets;
        float r; int dw, dh;
        
        while (running) {
            char c;
            if (recv(client_socket, &c, 1, MSG_DONTWAIT | MSG_PEEK) == 0) break;
            
            bool has_new = false;
            {
                std::unique_lock<std::mutex> lock(web_mutex, std::try_to_lock);
                if (lock.owns_lock() && web_data_buffer.valid) {
                    web_data_buffer.img.copyTo(local_frame);
                    local_dets = web_data_buffer.dets;
                    r = web_data_buffer.r; dw = web_data_buffer.dw; dh = web_data_buffer.dh;
                    has_new = true;
                }
            }

            if (has_new && !local_frame.empty()) {
                for (const auto& det : local_dets) {
                    if (det.class_id != 0) continue;
                    float x1 = std::max(0.0f, (det.x1 - dw) / r);
                    float y1 = std::max(0.0f, (det.y1 - dh) / r);
                    float x2 = std::min((float)local_frame.cols, (det.x2 - dw) / r);
                    float y2 = std::min((float)local_frame.rows, (det.y2 - dh) / r);
                    
                    cv::rectangle(local_frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                    float cx = (x1 + x2) / 2.0f;
                    float cy = y1 + (y2 - y1) * AIM_HEIGHT_RATIO;
                    cv::circle(local_frame, cv::Point(cx, cy), 4, cv::Scalar(0, 0, 255), -1);
                }
                if (cv::imencode(".jpg", local_frame, buf, params)) {
                    std::string ph = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + std::to_string(buf.size()) + "\r\n\r\n";
                    if (send(client_socket, ph.c_str(), ph.length(), MSG_NOSIGNAL) < 0) break;
                    if (send(client_socket, buf.data(), buf.size(), MSG_NOSIGNAL) < 0) break;
                    if (send(client_socket, "\r\n", 2, MSG_NOSIGNAL) < 0) break;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    } else {
        std::string r = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n"
                        "<html><body style='margin:0;background:#000;display:flex;justify-content:center;height:100vh;align-items:center;'>"
                        "<img src='/stream' style='max-width:100%;max-height:100%;'/></body></html>";
        send(client_socket, r.c_str(), r.length(), 0);
    }
    close(client_socket);
    active_connections--;
}

// --- Main Loop ---
int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    signal(SIGINT, [](int){ running = false; hid_cv.notify_all(); cap_state.cv.notify_all(); });
    signal(SIGPIPE, SIG_IGN);
    
    set_affinity_big(); 
    set_realtime_priority();
    lock_memory();
    hid_init();
    
    std::thread hid_thread([](){ set_affinity_little(); hid_worker(); });
    std::thread web_thread(web_worker);
    std::thread cleaner_thread([](){ // Background FD Scrubber
        set_affinity_little();
        while(running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            scrub_sync_files();
        }
    });
    cleaner_thread.detach(); 
    
    void* ctx = init_model(MODEL_PATH);
    if (!ctx) { std::cerr << "Model Init Failed" << std::endl; return -1; }

    v4l2_context_t camera_ctx;
    // Open Camera (Request 180hz)
    if (v4l2_open(&camera_ctx, "/dev/video20", 1920, 1080, 180) < 0) return -1;
    v4l2_start_stream(&camera_ctx);
    std::cout << "[System] Manual V4L2 Driver Started" << std::endl;
    
    std::thread cap_thread(capture_worker, &camera_ctx);
    
    cv::Mat letterbox_img(640, 640, CV_8UC3);
    cv::Mat rgb_img(640, 640, CV_8UC3);
    rgb_img.setTo(cv::Scalar(114, 114, 114));
    
    Detection dets[50];
    auto last_print_time = std::chrono::high_resolution_clock::now();
    double acc_total = 0, acc_pre = 0, acc_infer = 0;
    int frame_count = 0;
    
    const float cx = 1920 / 2.0f;
    const float cy = 1080 / 2.0f;

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
        float r; int dw, dh;
        
        r = std::min(640.0f/1080.0f, 640.0f/1920.0f);
        int nw = round(1920*r), nh = round(1080*r);
        dw = (640-nw)/2; dh = (640-nh)/2;

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

        // Web UI Clone (If needed)
        if (active_connections > 0 && web_mutex.try_lock()) {
             cv::Mat wrapper(1080, 1920, CV_8UC3, raw_data);
             web_data_buffer.img = wrapper.clone();
             web_data_buffer.dets.clear();
             web_data_buffer.valid = false;
             web_mutex.unlock();
        }

        v4l2_manual_qbuf(&camera_ctx, current_idx);
        auto t_pre_end = std::chrono::high_resolution_clock::now();

        // Inference
        int count = detect(ctx, rgb_img.data, CONF_THRES, NMS_THRES, dets, 50);
        auto t_infer_end = std::chrono::high_resolution_clock::now();

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
            float cd_x = (pid_state.tracking_frames > 0) ? (raw_dx - pid_state.last_error_x) : 0;
            float cd_y = (pid_state.tracking_frames > 0) ? (raw_dy - pid_state.last_error_y) : 0;
            
            pid_state.last_error_x = raw_dx; pid_state.last_error_y = raw_dy;
            pid_state.tracking_frames++;
            
            if (std::abs(raw_dx) < AIM_DEADZONE) raw_dx = 0;
            if (std::abs(raw_dy) < AIM_DEADZONE) raw_dy = 0;

            float mx = (PID_KP * raw_dx) + (PID_KD * cd_x);
            float my = (PID_KP * raw_dy) + (PID_KD * cd_y);
            
            {
                std::lock_guard<std::mutex> lock(hid_mutex);
                hid_buffer_x += mx * MOUSE_SENSITIVITY;
                hid_buffer_y += my * MOUSE_SENSITIVITY;
            }
            hid_cv.notify_one();
        } else {
            pid_state.tracking_frames = 0;
            std::lock_guard<std::mutex> lock(hid_mutex);
            hid_buffer_x = 0; hid_buffer_y = 0;
        }

        if (active_connections > 0) {
            std::lock_guard<std::mutex> lock(web_mutex);
            web_data_buffer.dets.assign(dets, dets + count);
            web_data_buffer.r = r; web_data_buffer.dw = dw; web_data_buffer.dh = dh;
            web_data_buffer.valid = true;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        acc_total += std::chrono::duration<double, std::milli>(t_end - t_start).count();
        acc_pre += std::chrono::duration<double, std::milli>(t_pre_end - t_start).count(); // includes read
        acc_infer += std::chrono::duration<double, std::milli>(t_infer_end - t_pre_end).count();
        frame_count++;

        if (std::chrono::duration<double>(t_end - last_print_time).count() >= 1.0) {
            std::cout << std::fixed << std::setprecision(2)
                      << "[FPS: " << frame_count << "] Total: " << (acc_total/frame_count) 
                      << "ms (Pre: " << (acc_pre/frame_count) << ", NPU: " << (acc_infer/frame_count) << ")" 
                      << "\r" << std::flush;
            last_print_time = t_end; acc_total=0; acc_pre=0; acc_infer=0; frame_count=0;
        }
    }
    
    running = false;
    if(hid_thread.joinable()) hid_thread.join();
    if(web_thread.joinable()) web_thread.join();
    return 0;
}
