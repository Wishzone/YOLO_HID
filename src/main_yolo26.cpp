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

// Forward Declarations
void set_affinity_little();
void set_affinity_big();

// --- 配置 ---
// YOLO26 修改后的模型路径
#define MODEL_PATH "./Models/cf-26n-rk3588-int8_100.rknn"

#define HID_DEVICE "/dev/hidg1"
#define CAMERA_INDEX 20
#define CONF_THRES 0.65f
#define NMS_THRES 0.45f
#define HTTP_PORT 8080
#define MJPEG_QUALITY 80 

// 控制参数
// 针对低延迟/即时响应进行了调优
const float PID_KP = 0.60f;             // 高 P 值用于即时反应
const float PID_KD = 0.35f;             // D 值用于抑制高 P 值带来的震荡
const float MOUSE_SENSITIVITY = 1.0f;   // 1:1 像素映射（如果太快请调整游戏内灵敏度）
const float HID_SMOOTH_FACTOR = 1.0f;   // 无平滑（立即施加力量）
const float AIM_DEADZONE = 0.0f; 
const float AIM_HEIGHT_RATIO = 0.20f;   // 0.2 = 头部水平（框顶部的 20%）
const float AUTO_FIRE_RADIUS = 15.0f;
const int MAX_DETECTIONS = 10;          // 最大检测目标数量
const int BURST_PRESS_MS = 25;          // 点射：按下时间 (ms)
const int BURST_GAP_MS = 50;            // 点射：间隔时间 (ms)

// --- 全局状态 ---
std::atomic<bool> running(true);
std::atomic<bool> fire_request(false);
std::atomic<int> active_connections(0);
int hid_fd = -1;

// HID 缓冲
float hid_buffer_x = 0.0f;
float hid_buffer_y = 0.0f;
std::mutex hid_mutex;
std::condition_variable hid_cv;

// Web 显示数据
// 使用 Detection26 结构体
struct WebData {
    cv::Mat img;
    std::vector<Detection26> dets;
    float r; int dw, dh;
    bool valid = false;
    uint64_t frame_id = 0;
} web_data_buffer;
std::mutex web_mutex;

// PID 状态
struct {
    float last_error_x = 0;
    float last_error_y = 0;
    int tracking_frames = 0;
} pid_state;

// --- Web 卸载处理逻辑 ---
std::atomic<int> web_transfer_buf_idx{-1};
std::mutex web_transfer_mutex;
std::vector<Detection26> web_transfer_dets;
// 这些参数用于 Web 线程做 RGA 处理
float web_transfer_r; 
int web_transfer_dw, web_transfer_dh;
int web_transfer_src_dma;
int web_transfer_src_w, web_transfer_src_h;
int web_transfer_length;
void* web_transfer_data = nullptr;

// 独立的 Web 帧处理线程（Consumer）
// 它从主线程接收 V4L2 Buffer 索引，执行 RGA Copy，更新 WebData，然后归还 Buffer。
void web_processor_func(v4l2_context_t* v4l2_ctx) {
    set_affinity_little(); // 在小核上运行，不影响推理
    
    while(running) {
        // 1. 获取 Buffer
        // 这是一个自旋等待还是休眠？由于我们希望“尽力而为”，休眠是好的。
        int idx = web_transfer_buf_idx.exchange(-1);
        if (idx == -1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        
        // 2. 拿到数据了，开始处理
        // 从全局 transfer 变量读取参数 (不需要太严格的锁，因为主线程只有在 web_transfer_buf_idx == -1 时才会写这些变量)
        // 但为了安全，我们可以加个轻量锁，或者依靠原子变量的 happens-before
        // 实际上，只要主线程遵循 protocol: "Write Params -> Set Atomic IDX"，Web 线程 "Read Atomic IDX -> Read Params"，内存顺序应该就是安全的。
        
        std::vector<Detection26> local_dets;
        {
            std::lock_guard<std::mutex> lock(web_transfer_mutex);
            local_dets = web_transfer_dets;
        }
        
        // 3. RGA Copy / Decode
        bool processed = false;
        
        // 准备目标 Web 缓冲区
        {
             // 只有在要写入时才锁定 Web 缓冲区
             // 我们先在本地做 RGA（如果有双缓冲更好，但这里我们可以直接写入 WebData 的 Mat，因为从 WebData 读取是瞬间的 memcpy 或 encode）
             // 为了减少 WebMutex 持有时间，我们可以暂时只申请内存
        }

        // 注意：我们必须直接往 WebData 写，或者写到一个本地 Mat 然后 swap。
        // 为了简单，我们直接尝试 lock web_mutex。如果 Web Server 正在发送（持有锁），我们稍等一下也没事。
        // 但为了极致性能，我们可以在这里使用双缓冲，但目前已经是“后台线程”了，稍微阻塞一下不影响主线程 FPS。
        
        std::unique_lock<std::mutex> lock(web_mutex);
        
        if (web_data_buffer.img.empty() || 
            web_data_buffer.img.cols != web_transfer_src_w || 
            web_data_buffer.img.rows != web_transfer_src_h) {
             web_data_buffer.img.create(web_transfer_src_h, web_transfer_src_w, CV_8UC3);
        }

        if (v4l2_ctx->format == V4L2_PIX_FMT_BGR24) {
             rga_buffer_t src_rga = wrapbuffer_fd(web_transfer_src_dma, web_transfer_src_w, web_transfer_src_h, RK_FORMAT_BGR_888);
             rga_buffer_t dst_rga = wrapbuffer_virtualaddr(web_data_buffer.img.data, web_transfer_src_w, web_transfer_src_h, RK_FORMAT_BGR_888);
             imcopy(src_rga, dst_rga);
        } else {
             // MJPEG Fallback
             cv::Mat raw_mat(1, web_transfer_length, CV_8UC1, web_transfer_data);
             cv::Mat decoded;
             cv::imdecode(raw_mat, cv::IMREAD_COLOR, &decoded);
             if(!decoded.empty()) decoded.copyTo(web_data_buffer.img);
        }
        
        // 4. 更新检测结果
        web_data_buffer.dets = local_dets;
        web_data_buffer.r = web_transfer_r;
        web_data_buffer.dw = web_transfer_dw;
        web_data_buffer.dh = web_transfer_dh;
        web_data_buffer.valid = true;
        web_data_buffer.frame_id++;
        
        lock.unlock(); // Done updating web data
        
        // 5. 归还 Buffer 给 V4L2 驱动 (关键！)
        v4l2_manual_qbuf(v4l2_ctx, idx);
    }
}

// --- 系统工具 ---

// 异步捕获状态（类似 main.cpp）
struct {
    std::mutex mutex;
    std::condition_variable cv;
    int latest_idx = -1;
    bool new_frame_available = false;
    v4l2_context_t* ctx = nullptr;
} cap_state;

// 独立的捕获线程，使用 SELECT 积极地排空缓冲区队列（类似 main.cpp）
void capture_worker_func(v4l2_context_t* ctx) {
    cap_state.ctx = ctx;
    
    // 在工作线程内设置 V4L2 缓冲区
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
        
        // 阻塞等待帧
        if (select(ctx->fd + 1, &fds, NULL, NULL, &tv) <= 0) continue;
        
        // 出队
        if (v4l2_manual_dqbuf(ctx, &buf) == 0) {
            {
                std::lock_guard<std::mutex> lock(cap_state.mutex);
                int old_idx = cap_state.latest_idx;
                
                // 如果有一帧旧数据尚未处理，立即归还给驱动（丢弃）
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

// --- HID 线程 ---
enum BurstState { BST_IDLE, BST_PRESS, BST_COOLDOWN };

void hid_thread_func() {
    set_affinity_little();
    hid_fd = open(HID_DEVICE, O_RDWR | O_NONBLOCK);
    if (hid_fd < 0) {
        std::cerr << "[HID] Failed to open " << HID_DEVICE << std::endl;
    }

    float remaining_x = 0.0f, remaining_y = 0.0f;
    char report[4] = {0}; 
    
    // 点射控制
    BurstState burst_state = BST_IDLE;
    auto last_burst_tick = std::chrono::steady_clock::now();
    // 点射参数现在是顶部的全局常量
    
    bool last_physical_click = false;

    while (running) {
        // 1. 获取输入逻辑
        {
            std::unique_lock<std::mutex> lock(hid_mutex);
            
            // 等待逻辑：
            // 如果请求开火 -> 不等待（需要循环以控制点射时序）
            // 如果有待处理的移动 -> 不等待
            // 如果空闲 -> 等待
            if (std::abs(remaining_x) < 0.1f && std::abs(remaining_y) < 0.1f && 
                hid_buffer_x == 0 && hid_buffer_y == 0 && 
                !fire_request) { // 如果 fire_request 为真，我们循环
                
                hid_cv.wait(lock, [&] { 
                    return !running || (hid_buffer_x != 0 || hid_buffer_y != 0 || fire_request); 
                });
            }
            
            if (!running) break;

            // 添加新输入
            if (hid_buffer_x != 0 || hid_buffer_y != 0) {
                remaining_x += hid_buffer_x;
                remaining_y += hid_buffer_y;
                hid_buffer_x = 0;
                hid_buffer_y = 0;
            }
        }

        // 2. 高频循环 (1000Hz)
        if (hid_fd >= 0) {
            // --- 点射逻辑 ---
            bool physical_click = false;
            auto now = std::chrono::steady_clock::now();
            
            if (fire_request) {
                // 点击的状态机
                if (burst_state == BST_IDLE) {
                    burst_state = BST_PRESS;
                    last_burst_tick = now;
                    physical_click = true;
                } else if (burst_state == BST_PRESS) {
                    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_burst_tick).count() >= BURST_PRESS_MS) {
                        burst_state = BST_COOLDOWN;
                        last_burst_tick = now;
                        physical_click = false;
                    } else {
                        physical_click = true;
                    }
                } else if (burst_state == BST_COOLDOWN) {
                    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_burst_tick).count() >= BURST_GAP_MS) {
                        burst_state = BST_IDLE; 
                        // 将在下一次迭代中触发按下
                        physical_click = false; 
                    } else {
                        physical_click = false;
                    }
                }
            } else {
                burst_state = BST_IDLE;
                physical_click = false;
            }

            // 立即应用所有剩余移动量（受 USB 限制）
            float step_x = remaining_x;
            float step_y = remaining_y;
            
            int dx = static_cast<int>(std::round(step_x));
            int dy = static_cast<int>(std::round(step_y));
            
            dx = std::max(-127, std::min(127, dx));
            dy = std::max(-127, std::min(127, dy));
            
            if (dx != 0 || dy != 0 || physical_click != last_physical_click) {
                report[0] = physical_click ? 0x01 : 0; 
                report[1] = (char)dx;
                report[2] = (char)dy;
                report[3] = 0;
                
                if (write(hid_fd, report, 4) < 0) {
                     // 忽略
                }
                
                remaining_x -= dx;
                remaining_y -= dy;
                last_physical_click = physical_click;
            }
        }
        
        // 轮询率 ~1000Hz 以尽可能快地清空队列
        std::this_thread::sleep_for(std::chrono::microseconds(800));
    }
    if (hid_fd >= 0) close(hid_fd);
}

// --- HTTP 服务器 ---
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
                if (active_connections >= 5) {
                    close(new_socket);
                    continue;
                }
                active_connections++;
                std::thread([new_socket]() {
                    struct timeval tv = {3, 0};
                    setsockopt(new_socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
                    setsockopt(new_socket, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

                    const char* header = "HTTP/1.1 200 OK\r\n"
                                       "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
                    send(new_socket, header, strlen(header), 0);
                    
                    uint64_t last_frame_id = 0;
                    std::vector<uchar> buf;
                    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, MJPEG_QUALITY};
                    
                    while (running && active_connections > 0) {
                        cv::Mat frame;
                        std::vector<Detection26> local_dets;
                        float local_r; int local_dw, local_dh;
                        bool has_new_data = false;
                        
                        {
                            std::unique_lock<std::mutex> lock(web_mutex, std::try_to_lock); // 使用 try_lock 避免阻塞推理线程
                            if (lock.owns_lock() && web_data_buffer.valid && !web_data_buffer.img.empty()) {
                                if (web_data_buffer.frame_id > last_frame_id) {
                                    frame = web_data_buffer.img.clone();
                                    local_dets = web_data_buffer.dets;
                                    local_r = web_data_buffer.r;
                                    local_dw = web_data_buffer.dw;
                                    local_dh = web_data_buffer.dh;
                                    last_frame_id = web_data_buffer.frame_id;
                                    has_new_data = true;
                                }
                            }
                        }

                        if (!has_new_data) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(2)); // 短暂休眠以避免空转
                            continue;
                        }
                            
                        // Visualization (Outside lock)
                        for (const auto& det : local_dets) {
                            int x1 = (det.x1 - local_dw) / local_r;
                            int y1 = (det.y1 - local_dh) / local_r;
                            int x2 = (det.x2 - local_dw) / local_r;
                            int y2 = (det.y2 - local_dh) / local_r;
                            
                            cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                            std::string label = std::to_string(det.class_id) + " " + 
                                                std::to_string((int)(det.score * 100)) + "%";
                            cv::putText(frame, label, cv::Point(x1, y1 - 5), 
                                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
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
                    }
                    close(new_socket);
                    active_connections--;
                }).detach();
            }
        }
    }
    close(server_fd);
}

// --- 主循环 ---
int main(int argc, char** argv) {
    int target_class_id = 0;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-QF") {
            target_class_id = 1;
            std::cout << "[System] QF Mode Enabled: Targeting Class 1" << std::endl;
        }
    }

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    cv::setNumThreads(1); // [关键修复] 禁止 OpenCV 创建额外的工作线程（防止它们跑去大核）
    signal(SIGPIPE, SIG_IGN); // 忽略 SIGPIPE，防止 Socket 关闭导致程序退出
    set_realtime_priority();
    lock_memory();
    set_affinity_big();

    // 初始化模型
    void* rknn_ctx = init_yolo26_model(MODEL_PATH);
    if (!rknn_ctx) return -1;

    // 初始化摄像头
    v4l2_context_t v4l2_ctx;
    char cam_path[64];
    snprintf(cam_path, sizeof(cam_path), "/dev/video%d", CAMERA_INDEX);
    if (v4l2_open(&v4l2_ctx, cam_path, 640, 480, 30) < 0) {
        std::cerr << "[V4L2] Init failed" << std::endl;
        release_yolo26_model(rknn_ctx);
        return -1;
    }
    v4l2_start_stream(&v4l2_ctx);

    // 线程
    std::thread t_hid(hid_thread_func);
    std::thread t_http(http_server_func);
    std::thread t_web_proc(web_processor_func, &v4l2_ctx); // 启动 Web 帧处理线程

    std::thread cleaner_thread([](){ 
        set_affinity_little();
        while(running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            scrub_sync_files();
        }
    });
    cleaner_thread.detach();

    // 缓冲区
    const int model_width = 640;
    const int model_height = 640;
    cv::Mat bgr_frame; // 仅在需要时分配
    
    // 分配 RKNN 输入缓冲区（首选 Pinned/Aligned，但 malloc 也适用于 RGA 虚拟地址）
    // 我们使用 vector 为了简单，确保内存连续
    std::vector<unsigned char> rknn_input_mem(model_width * model_height * 3); /* RGB888 */
    
    Detection26 results[MAX_DETECTIONS];
    
    struct v4l2_buffer buf;
    struct v4l2_plane planes[8]; // For MPLANE support
    memset(&buf, 0, sizeof(buf));
    if(v4l2_ctx.is_mplane) buf.m.planes = planes;

    // 启动捕获线程
    std::thread cap_thread(capture_worker_func, &v4l2_ctx);
    cap_thread.detach();

    auto t_last_print = std::chrono::steady_clock::now();

    // 预先计算 Letterbox 参数
    int src_w = v4l2_ctx.width;
    int src_h = v4l2_ctx.height;
    float r_ratio = std::min((float)model_width / src_w, (float)model_height / src_h);
    int nw = src_w * r_ratio;
    int nh = src_h * r_ratio;
    int dw = (model_width - nw) / 2;
    int dh = (model_height - nh) / 2;

    while (running) {
        auto t_start = std::chrono::steady_clock::now();

        // 1. 捕获 - 等待线程发来的最新帧
        int buf_idx = -1;
        {
            std::unique_lock<std::mutex> lock(cap_state.mutex);
            cap_state.cv.wait(lock, []{ return cap_state.new_frame_available || !running; });
            if (!running) break;
            
            buf_idx = cap_state.latest_idx;
            cap_state.latest_idx = -1; // 我们取走了它
            cap_state.new_frame_available = false;
        }

        if (buf_idx == -1) continue; 
        
        // 恢复当前索引的缓冲区元数据
        buf.index = buf_idx;
        buf.type = v4l2_ctx.type;
        buf.memory = V4L2_MEMORY_MMAP;
        
        // 为了零拷贝优化，我们需要 dma_fd
        int src_dma_fd = v4l2_ctx.buffers[buf_idx].dma_fd;
        void* frame_data = v4l2_ctx.buffers[buf_idx].start;
        int length = v4l2_ctx.buffers[buf_idx].length;
        
        // 2. 使用 RGA 进行预处理（硬件缩放）
        rga_buffer_t src_rga;
        rga_buffer_t dst_rga;
        memset(&src_rga, 0, sizeof(src_rga));
        memset(&dst_rga, 0, sizeof(dst_rga));
        
        src_rga = wrapbuffer_fd(src_dma_fd, src_w, src_h, RK_FORMAT_BGR_888);
        
        // 指针偏移到 (dw, dh)
        unsigned char* dst_ptr = rknn_input_mem.data() + (dh * model_width + dw) * 3;
        dst_rga = wrapbuffer_virtualaddr(dst_ptr, nw, nh, RK_FORMAT_RGB_888);
        dst_rga.wstride = model_width; 
        
        imresize(src_rga, dst_rga);
        
        // 3. 推理
        int count = detect_yolo26(
            rknn_ctx, 
            rknn_input_mem.data(), 
            CONF_THRES, 
            NMS_THRES, 
            results, 
            MAX_DETECTIONS
        );
        
        // 4. 逻辑与更新状态
        float cx = model_width / 2.0f;
        float cy = model_height / 2.0f;
        float min_dist = 10000.0f;
        Detection26* target = nullptr;

        for (int i = 0; i < count; i++) {
            if (results[i].class_id != target_class_id) continue;


            // 寻找离中心最近的目标
            float bx = (results[i].x1 + results[i].x2) / 2.0f;
            float by = (results[i].y1 + results[i].y2) / 2.0f;
            float dist = sqrt(pow(bx - cx, 2) + pow(by - cy, 2));
            if (dist < min_dist) {
                min_dist = dist;
                target = &results[i];
            }
        }

        if (target) {
            float box_h = target->y2 - target->y1;
            float bx = (target->x1 + target->x2) / 2.0f;
            // 修复：使用 AIM_HEIGHT_RATIO 瞄准头部 (y1 + h * 0.2) 而不是缩放误差
            float by = target->y1 + box_h * AIM_HEIGHT_RATIO;
            
            // PID
            float err_x = (bx - cx); 
            float err_y = (by - cy);
            
            // 应用死区（如果有）
            if (std::abs(err_x) < AIM_DEADZONE) err_x = 0;
            if (std::abs(err_y) < AIM_DEADZONE) err_y = 0;
            
            float out_x = err_x * PID_KP + (err_x - pid_state.last_error_x) * PID_KD;
            float out_y = err_y * PID_KP + (err_y - pid_state.last_error_y) * PID_KD;
            
            pid_state.last_error_x = err_x;
            pid_state.last_error_y = err_y;

            // 自动开火逻辑（预测性）
            // 如果在半径内或者下一帧可能命中（预测），则开火
            // PID_KP 为 0.8，意味着我们在 1 帧内修正 80% 的误差。
            // 如果当前误差是 50，下一帧误差约 10。10 在 15 以内。所以现在就开火。
            float predicted_dist = sqrt(pow(err_x * (1.0f - PID_KP), 2) + pow(err_y * (1.0f - PID_KP), 2));
            
            // 安全上限：仅当我们偏离不太远时使用预测（例如 2.5 倍半径）
            // 这防止了“粘滞开火”，即目标很远但由于滞后或过冲，预测认为“足够近”，导致在严重偏离时仍扣住扳机。
            bool safe_to_predict = min_dist < (AUTO_FIRE_RADIUS * 2.5f);
            
            bool should_fire = (min_dist < AUTO_FIRE_RADIUS) || (predicted_dist < AUTO_FIRE_RADIUS && safe_to_predict);
            
            if (fire_request != should_fire) {
                fire_request = should_fire;
                hid_cv.notify_one();
            }
            
            {
                std::lock_guard<std::mutex> lock(hid_mutex);
                hid_buffer_x += out_x * MOUSE_SENSITIVITY;
                hid_buffer_y += out_y * MOUSE_SENSITIVITY;
            }
            hid_cv.notify_one();
        } else {
             // 跟踪丢失时重置 PID 状态，以防止下次获取时“跳变”
             pid_state.last_error_x = 0;
             pid_state.last_error_y = 0;

             if (fire_request) {
                 fire_request = false;
                 hid_cv.notify_one();
             }
        }


        // 5. 将 Buffer 转让给 Web 线程 (如果它空闲)
        // 只有当有活跃连接时才尝试转让，否则直接归还
        bool buffer_offloaded = false;
        
        if (active_connections > 0) {
            int expected = -1;
            // 尝试原子地将 web_transfer_buf_idx 从 -1 改为 buf_idx
            // 如果它不是 -1 (说明 Web 线程还在忙上一帧)，则 compare_exchange 失败，我们不做 offload。
            // 在修改 idx 之前，先准备好参数 (这在单生产者-单消费者下是安全的，因为消费者只在 idx != -1 时读)
            if (std::atomic_load(&web_transfer_buf_idx) == -1) {
                // 填充参数
                {
                    std::lock_guard<std::mutex> lock(web_transfer_mutex);
                    web_transfer_dets.clear();
                    for(int i=0; i<count; i++) web_transfer_dets.push_back(results[i]);
                }
                web_transfer_r = r_ratio;
                web_transfer_dw = dw;
                web_transfer_dh = dh;
                web_transfer_src_dma = src_dma_fd;
                web_transfer_src_w = src_w;
                web_transfer_src_h = src_h;
                web_transfer_length = length;
                web_transfer_data = frame_data; // 只要 buffer 没归还，这个指针就是有效的

                // 提交任务
                if (web_transfer_buf_idx.compare_exchange_strong(expected, buf_idx)) {
                    buffer_offloaded = true;
                }
            }
        }
        
        // 如果没有成功转让给 Web 线程 (因为 Web 忙 或者 无连接)，必须立即归还 Buffer
        if (!buffer_offloaded) {
            v4l2_manual_qbuf(&v4l2_ctx, buf_idx);
        }

        auto t_end = std::chrono::steady_clock::now();
        
        if (std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_last_print).count() >= 1000) {
            double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            
            std::cout << "\r[System] Running | FPS: " << std::fixed << std::setprecision(1) << 1000.0 / total_ms 
                      << "   " << std::flush;
            t_last_print = t_end;
        }
    }

    running = false;
    hid_cv.notify_all();
    t_hid.join();
    t_http.join();
    if(t_web_proc.joinable()) t_web_proc.join();
    
    release_yolo26_model(rknn_ctx);
    v4l2_close(&v4l2_ctx);
    
    return 0;
}
