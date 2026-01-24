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
#include <rga/RgaApi.h>
#include <sys/ioctl.h>
#include "yolo26_rknn.h"
#include "v4l2_driver.h"

// --- Configuration ---
// Modified Model Path for YOLO26
#define MODEL_PATH "./Models/cf-26n-rk3588-int8_80.rknn"

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
// Use Detection26 struct
struct WebData {
    cv::Mat img;
    std::vector<Detection26> dets;
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

// Async Capture State (Similar to main.cpp)
struct {
    std::mutex mutex;
    std::condition_variable cv;
    int latest_idx = -1;
    bool new_frame_available = false;
    v4l2_context_t* ctx = nullptr;
} cap_state;

// Separate thread for capturing to drain buffer queue aggressively using SELECT (like main.cpp)
void capture_worker_func(v4l2_context_t* ctx) {
    cap_state.ctx = ctx;
    
    // Setup V4L2 Buffers inside worker
    struct v4l2_buffer buf;
    struct v4l2_plane planes[8];
    memset(&buf, 0, sizeof(buf));
    if(ctx->is_mplane) buf.m.planes = planes;

    while (running) {
        buf.type = ctx->type;
        buf.memory = V4L2_MEMORY_MMAP;
        
        fd_set fds;
        struct timeval tv;
        FD_ZERO(&fds);
        FD_SET(ctx->fd, &fds);
        tv.tv_sec = 1; tv.tv_usec = 0;
        
        // Blocking wait for frame
        if (select(ctx->fd + 1, &fds, NULL, NULL, &tv) <= 0) continue;
        
        // Dequeue
        if (v4l2_manual_dqbuf(ctx, &buf) == 0) {
            {
                std::lock_guard<std::mutex> lock(cap_state.mutex);
                int old_idx = cap_state.latest_idx;
                
                // If there was an old frame not processed yet, return it to driver immediately (Drop)
                if (old_idx != -1) {
                    v4l2_manual_qbuf(ctx, old_idx);
                }
                
                cap_state.latest_idx = buf.index;
                cap_state.new_frame_available = true;
            }
            cap_state.cv.notify_one();
        }
    }
}


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

// --- HID Thread ---
void hid_thread_func() {
    set_affinity_little();
    hid_fd = open(HID_DEVICE, O_RDWR | O_NONBLOCK);
    if (hid_fd < 0) {
        std::cerr << "[HID] Failed to open " << HID_DEVICE << std::endl;
    }

    float current_x = 0.0f, current_y = 0.0f;
    char report[5] = {0}; 

    while (running) {
        std::unique_lock<std::mutex> lock(hid_mutex);
        hid_cv.wait(lock, [] { return !running || (hid_buffer_x != 0 || hid_buffer_y != 0); });
        
        if (!running) break;

        current_x = current_x * (1.0f - HID_SMOOTH_FACTOR) + hid_buffer_x * HID_SMOOTH_FACTOR;
        current_y = current_y * (1.0f - HID_SMOOTH_FACTOR) + hid_buffer_y * HID_SMOOTH_FACTOR;
        
        hid_buffer_x = 0; hid_buffer_y = 0;
        lock.unlock();

        if (hid_fd >= 0) {
            int dx = static_cast<int>(std::round(current_x));
            int dy = static_cast<int>(std::round(current_y));
            
            if (dx != 0 || dy != 0) {
                dx = std::max(-127, std::min(127, dx));
                dy = std::max(-127, std::min(127, dy));
                
                report[0] = 0x01; 
                report[1] = 0;    
                report[2] = (char)dx;
                report[3] = (char)dy;
                report[4] = 0;    
                
                write(hid_fd, report, 5);
            }
        }
    }
    if (hid_fd >= 0) close(hid_fd);
}

// --- HTTP Server ---
void http_server_func() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
    
    sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(HTTP_PORT);
    
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "[Web] Bind failed" << std::endl;
        return;
    }
    listen(server_fd, 3);
    std::cout << "[Web] Streaming on port " << HTTP_PORT << std::endl;

    while (running) {
        struct pollfd pfd = {server_fd, POLLIN, 0};
        if (poll(&pfd, 1, 1000) > 0) {
            int new_socket = accept(server_fd, nullptr, nullptr);
            if (new_socket >= 0) {
                active_connections++;
                std::thread([new_socket]() {
                    const char* header = "HTTP/1.1 200 OK\r\n"
                                       "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
                    send(new_socket, header, strlen(header), 0);
                    
                    auto last_frame_time = std::chrono::steady_clock::now();
                    std::vector<uchar> buf;
                    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, MJPEG_QUALITY};
                    
                    while (running && active_connections > 0) {
                        auto now = std::chrono::steady_clock::now();
                        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_frame_time).count() < 1000/WEB_MAX_FPS) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                            continue;
                        }

                        cv::Mat frame;
                        std::vector<Detection26> dets;
                        
                        {
                            std::unique_lock<std::mutex> lock(web_mutex);
                            if (!web_data_buffer.valid || web_data_buffer.img.empty()) {
                                lock.unlock();
                                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                                continue;
                            }
                            frame = web_data_buffer.img.clone();
                            dets = web_data_buffer.dets;
                            
                            // Visualization
                            for (const auto& det : dets) {
                                int x1 = (det.x1 - web_data_buffer.dw) / web_data_buffer.r;
                                int y1 = (det.y1 - web_data_buffer.dh) / web_data_buffer.r;
                                int x2 = (det.x2 - web_data_buffer.dw) / web_data_buffer.r;
                                int y2 = (det.y2 - web_data_buffer.dh) / web_data_buffer.r;
                                
                                cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                                std::string label = std::to_string(det.class_id) + " " + 
                                                  std::to_string((int)(det.score * 100)) + "%";
                                cv::putText(frame, label, cv::Point(x1, y1 - 5), 
                                          cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                            }
                        }

                        cv::imencode(".jpg", frame, buf, params);
                        std::string part_header = "--frame\r\n"
                                                "Content-Type: image/jpeg\r\n"
                                                "Content-Length: " + std::to_string(buf.size()) + "\r\n\r\n";
                        
                        if (send(new_socket, part_header.c_str(), part_header.length(), MSG_NOSIGNAL) < 0 ||
                            send(new_socket, buf.data(), buf.size(), MSG_NOSIGNAL) < 0 ||
                            send(new_socket, "\r\n", 2, MSG_NOSIGNAL) < 0) {
                            break;
                        }
                        
                        last_frame_time = now;
                    }
                    close(new_socket);
                    active_connections--;
                }).detach();
            }
        }
    }
    close(server_fd);
}

// --- Main Loop ---
int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    set_realtime_priority();
    lock_memory();
    set_affinity_big();

    // Init Model
    void* rknn_ctx = init_yolo26_model(MODEL_PATH);
    if (!rknn_ctx) return -1;

    // Init Camera
    v4l2_context_t v4l2_ctx;
    char cam_path[64];
    snprintf(cam_path, sizeof(cam_path), "/dev/video%d", CAMERA_INDEX);
    if (v4l2_open(&v4l2_ctx, cam_path, 640, 480, 30) < 0) {
        std::cerr << "[V4L2] Init failed" << std::endl;
        release_yolo26_model(rknn_ctx);
        return -1;
    }
    v4l2_start_stream(&v4l2_ctx);

    // Threads
    std::thread t_hid(hid_thread_func);
    std::thread t_http(http_server_func);

    // Buffers
    const int model_width = 640;
    const int model_height = 640;
    cv::Mat bgr_frame; // Only allocate when needed
    
    // Allocated RKNN Input Buffer (Pinned/Aligned preferable, but malloc works for RGA virtual address)
    // We use a vector for simplicity, ensures contiguous
    std::vector<unsigned char> rknn_input_mem(model_width * model_height * 3); /* RGB888 */
    
    #ifndef MAX_DETECTIONS
    #define MAX_DETECTIONS 100
    #endif
    Detection26 results[MAX_DETECTIONS];
    
    std::cout << "[System] Loop Start" << std::endl;
    
    struct v4l2_buffer buf;
    struct v4l2_plane planes[8]; // For MPLANE support
    memset(&buf, 0, sizeof(buf));
    if(v4l2_ctx.is_mplane) buf.m.planes = planes;

    // Start Capture Thread
    std::thread cap_thread(capture_worker_func, &v4l2_ctx);
    cap_thread.detach();

    auto t_last_print = std::chrono::steady_clock::now();

    while (running) {
        auto t_start = std::chrono::steady_clock::now();

        // 1. Capture - Wait for latest frame from thread
        int buf_idx = -1;
        {
            std::unique_lock<std::mutex> lock(cap_state.mutex);
            cap_state.cv.wait(lock, []{ return cap_state.new_frame_available || !running; });
            if (!running) break;
            
            buf_idx = cap_state.latest_idx;
            cap_state.latest_idx = -1; // We took it
            cap_state.new_frame_available = false;
        }

        if (buf_idx == -1) continue; 
        
        // Recover buffer metadata for current index
        buf.index = buf_idx;
        buf.type = v4l2_ctx.type;
        buf.memory = V4L2_MEMORY_MMAP;
        // Note: We don't have the full buf struct filled by DQBUF here, but we only need index 
        // and we know the rest from v4l2_ctx state or it's static
        
        // For 0-copy optimization, we need dma_fd
        int src_dma_fd = v4l2_ctx.buffers[buf_idx].dma_fd;
        void* frame_data = v4l2_ctx.buffers[buf_idx].start;
        int length = v4l2_ctx.buffers[buf_idx].length;
        int src_w = v4l2_ctx.width;
        int src_h = v4l2_ctx.height;
        
        auto t_cap = std::chrono::steady_clock::now();

        // 2. Preprocess using RGA (Hardware Resize)
        
        // Calculate Letterbox parameters
        float r = std::min((float)model_width / src_w, (float)model_height / src_h);
        int nw = src_w * r;
        int nh = src_h * r;
        int dw = (model_width - nw) / 2;
        int dh = (model_height - nh) / 2;

        // Reset background to gray (114)
        // This is fast enough on CPU for 640x640 (approx 0.3ms) or use RGA fill
        memset(rknn_input_mem.data(), 114, rknn_input_mem.size());
        
        // RGA Resize & Convert
        // Source: V4L2 DMA FD (BGR888 typically for V4L2_PIX_FMT_BGR24)
        // Destination: Virtual Address (RGB888 for RKNN)
        // We map destination to the ROI (Letterbox center)
        
        rga_buffer_t src_rga;
        rga_buffer_t dst_rga;
        memset(&src_rga, 0, sizeof(src_rga));
        memset(&dst_rga, 0, sizeof(dst_rga));
        
        src_rga = wrapbuffer_fd(src_dma_fd, src_w, src_h, RK_FORMAT_BGR_888);
        
        // Offset pointer to (dw, dh)
        unsigned char* dst_ptr = rknn_input_mem.data() + (dh * model_width + dw) * 3;
        dst_rga = wrapbuffer_virtualaddr(dst_ptr, nw, nh, RK_FORMAT_RGB_888);
        
        // Fix stride issue: wrapbuffer_virtualaddr implicitly sets stride = width
        // But our sub-buffer is part of a larger model_width line stride.
        // We must manually set the stride to model_width
        dst_rga.wstride = model_width; 
        
        // Execute RGA
        imresize(src_rga, dst_rga);
        
        // Web Display Handling (Lazy Copy)
        if (active_connections > 0) {
             // We need bgr_frame for web
             // Clone only if clients exist
             if (v4l2_ctx.format == V4L2_PIX_FMT_BGR24) {
                 cv::Mat bgr_raw(src_h, src_w, CV_8UC3, frame_data);
                 bgr_frame = bgr_raw.clone(); 
             } else {
                 // MJPEG fallback
                 cv::Mat raw_mat(1, length, CV_8UC1, frame_data);
                 cv::imdecode(raw_mat, cv::IMREAD_COLOR, &bgr_frame);
             }
        }
        
        // Release buffer immediately (Zero Copy for RGA inference path is done, 
        // but if Web used clone, we are safe too)
        v4l2_manual_qbuf(&v4l2_ctx, buf_idx); // <--- Return the buffer we processed

        auto t_pre = std::chrono::steady_clock::now();

        // 3. Inference
        int count = detect_yolo26(
            rknn_ctx, 
            rknn_input_mem.data(), 
            CONF_THRES, 
            NMS_THRES, 
            results, 
            MAX_DETECTIONS
        );
        
        auto t_infer = std::chrono::steady_clock::now();

        // 4. Logic & Update State
        float cx = model_width / 2.0f;
        float cy = model_height / 2.0f;
        float min_dist = 10000.0f;
        Detection26* target = nullptr;

        for (int i = 0; i < count; i++) {
            // Find closest to center
            float bx = (results[i].x1 + results[i].x2) / 2.0f;
            float by = (results[i].y1 + results[i].y2) / 2.0f;
            float dist = sqrt(pow(bx - cx, 2) + pow(by - cy, 2));
            if (dist < min_dist) {
                min_dist = dist;
                target = &results[i];
            }
        }

        if (target) {
            float bx = (target->x1 + target->x2) / 2.0f;
            float by = (target->y1 + target->y2) / 2.0f;
            
            // PID
            float err_x = (bx - cx) + AIM_DEADZONE; // Simple logic
            float err_y = (by - cy) * AIM_HEIGHT_RATIO;
            
            float out_x = err_x * PID_KP + (err_x - pid_state.last_error_x) * PID_KD;
            float out_y = err_y * PID_KP + (err_y - pid_state.last_error_y) * PID_KD;
            
            pid_state.last_error_x = err_x;
            pid_state.last_error_y = err_y;
            
            {
                std::lock_guard<std::mutex> lock(hid_mutex);
                hid_buffer_x += out_x * MOUSE_SENSITIVITY;
                hid_buffer_y += out_y * MOUSE_SENSITIVITY;
            }
            hid_cv.notify_one();
        }

        // 5. Update Web
        if (active_connections > 0) {
            std::lock_guard<std::mutex> lock(web_mutex);
            web_data_buffer.img = bgr_frame.clone(); // Original frame
            
            // Map detections back to orignal frame coordinates for display
            web_data_buffer.dets.clear();
            for (int i = 0; i < count; i++) {
                // Coordinates are in model space (640x640 with padding)
                // Need to be stored as is, and web thread will visualize using 'dw', 'dh', 'r'
                web_data_buffer.dets.push_back(results[i]);
            }
            web_data_buffer.r = r;
            web_data_buffer.dw = dw;
            web_data_buffer.dh = dh;
            web_data_buffer.valid = true;
        }

        auto t_end = std::chrono::steady_clock::now();
        
        if (std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_last_print).count() >= 200) {
            double cap_ms = std::chrono::duration<double, std::milli>(t_cap - t_start).count();
            double pre_ms = std::chrono::duration<double, std::milli>(t_pre - t_cap).count();
            double infer_ms = std::chrono::duration<double, std::milli>(t_infer - t_pre).count();
            double post_ms = std::chrono::duration<double, std::milli>(t_end - t_infer).count();
            double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            
            std::cout << "\r[Time] Cap: " << std::fixed << std::setprecision(2) << cap_ms << "ms | "
                      << "Pre: " << pre_ms << "ms | "
                      << "Infer: " << infer_ms << "ms | "
                      << "Post: " << post_ms << "ms | "
                      << "Total: " << total_ms << "ms | "
                      << "FPS: " << 1000.0 / total_ms 
                      << std::flush;
            t_last_print = t_end;
        }
    }

    running = false;
    hid_cv.notify_all();
    t_hid.join();
    t_http.join();
    
    release_yolo26_model(rknn_ctx);
    v4l2_close(&v4l2_ctx);
    
    return 0;
}
