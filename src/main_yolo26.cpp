#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
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
#include "modules/target_tracker.h"

// --- Configuration ---
#define MODEL_PATH "./Models/cs2-26n_rk3588_int8.rknn"
#define HID_DEVICE "/dev/hidg1"
#define CAMERA_INDEX 20
#define CONF_THRES 0.45f
#define HTTP_PORT 8080
#define MJPEG_QUALITY 80 

// Control Parameters
const float PID_KP = 0.60f;
const float PID_KD = 0.35f;
const float MOUSE_SENSITIVITY = 0.5f;
const float HID_SMOOTH_FACTOR = 1.0f;
const float AIM_DEADZONE = 0.0f; 
const float AIM_HEIGHT_RATIO = 0.20f;
const float AUTO_FIRE_RADIUS = 15.0f; 
const int MAX_DETECTIONS = 100;

// PID State
struct {
    float last_error_x = 0;
    float last_error_y = 0;
    int tracking_frames = 0;
} pid_state;

static bool parse_classes(const std::string& value, std::vector<int>& ids, const char* unrestricted) {
    ids.clear();
    if (value == unrestricted) return true;
    if (value.empty() || value.back() == ',') return false;
    std::istringstream input(value);
    std::string token;
    while (std::getline(input,token,',')) {
        if (token.empty() || token.find_first_not_of("0123456789") != std::string::npos) return false;
        try { ids.push_back(std::stoi(token)); } catch (...) { return false; }
    }
    return !ids.empty();
}
static bool parse_float(const std::string& value, float& number, float minimum, float maximum) {
    try {
        size_t consumed = 0;
        number = std::stof(value,&consumed);
        return consumed == value.size() && std::isfinite(number) && number >= minimum && number <= maximum;
    } catch (...) { return false; }
}

// --- Main Loop ---
int main(int argc, char* argv[]) {
    std::string model_path = MODEL_PATH;
    bool check_model = false;
    int scores_mode = 0;
    float nms_iou = 0.5f;
    std::string hid_device = HID_DEVICE;
    bool hid_enabled = true;
    std::vector<int> target_classes{0};
    TrackerOptions tracking_options;
    AimMode selected_mode = AimMode::Nearest;
    float hid_smoothing = HID_SMOOTH_FACTOR;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: " << argv[0] << " [--model PATH] [--scores auto|logits|probabilities] [--check-model]\n"
                      << "  [--hid-device PATH] [--no-hid] [--start-paused] [--target-classes all|0,1,...]\n"
                      << "  [--nms-iou VALUE] Raw-head duplicate suppression IoU, default 0.5 (0 < VALUE < 1).\n"
                      << "  [--aim-mode nearest|head-priority|head-only|body-only] [--head-classes none|0,1,...]\n"
                      << "  [--aim-radius PIXELS] [--prediction-ms 0..40] [--hid-smoothing 0.01..1]\n"
                      << "Defaults: nearest, no assumed head classes, full-frame radius, prediction off, smoothing 1.\n"
                      << "Default model: " << MODEL_PATH << "\n"
                      << "--check-model runs one NPU inference without camera, Web or HID.\n";
            return 0;
        } else if (argument == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (argument == "--check-model") {
            check_model = true;
        } else if (argument == "--no-hid") {
            hid_enabled = false;
        } else if (argument == "--start-paused") {
            hid_control_enabled = false;
        } else if (argument == "--hid-device" && i + 1 < argc) {
            hid_device = argv[++i];
        } else if (argument == "--target-classes" && i + 1 < argc) {
            const std::string classes = argv[++i];
            if (!parse_classes(classes,target_classes,"all")) { std::cerr << "Invalid target classes\n"; return 2; }
        } else if (argument == "--head-classes" && i + 1 < argc) {
            if (!parse_classes(argv[++i],tracking_options.head_classes,"none")) { std::cerr << "Invalid head classes\n"; return 2; }
        } else if (argument == "--aim-mode" && i + 1 < argc) {
            if (!parse_aim_mode(argv[++i],selected_mode)) { std::cerr << "Invalid aim mode\n"; return 2; }
        } else if ((argument == "--aim-radius" || argument == "--prediction-ms" || argument == "--hid-smoothing") && i + 1 < argc) {
            const std::string value = argv[++i];
            float* destination = argument == "--aim-radius" ? &tracking_options.aim_radius :
                                 argument == "--prediction-ms" ? &tracking_options.prediction_ms : &hid_smoothing;
            const float minimum = argument == "--hid-smoothing" ? 0.01f : 0;
            const float maximum = argument == "--hid-smoothing" ? 1 : argument == "--prediction-ms" ? 40 : 100000;
            if (!parse_float(value,*destination,minimum,maximum)) { std::cerr << "Invalid " << argument << '\n'; return 2; }
        } else if (argument == "--nms-iou" && i + 1 < argc) {
            const std::string value = argv[++i];
            try {
                size_t consumed = 0;
                nms_iou = std::stof(value, &consumed);
                if (consumed != value.size() || !std::isfinite(nms_iou) || nms_iou <= 0 || nms_iou >= 1)
                    throw std::invalid_argument("IoU");
            } catch (...) { std::cerr << "Invalid NMS IoU: " << value << std::endl; return 2; }
        } else if (argument == "--scores" && i + 1 < argc) {
            const std::string mode = argv[++i];
            if (mode == "auto") scores_mode = 0;
            else if (mode == "logits") scores_mode = 1;
            else if (mode == "probabilities") scores_mode = 2;
            else { std::cerr << "Unknown score mode: " << mode << "\n"; return 2; }
        } else {
            std::cerr << "Invalid or incomplete argument: " << argument << "\n";
            return 2;
        }
    }

    if (selected_mode != AimMode::Nearest && tracking_options.head_classes.empty()) {
        std::cerr << "This aim mode requires --head-classes from your model's labels\n"; return 2;
    }
    tracking_options.target_classes = target_classes;
    tracking_options.body_aim_ratio = AIM_HEIGHT_RATIO;
    aim_mode = static_cast<int>(selected_mode);
    aim_head_classes_available = !tracking_options.head_classes.empty();
    TargetTracker tracker(tracking_options);
    uint64_t last_control_revision = aim_settings_revision.load();

    std::cout << "[Control] Target classes:";
    if (target_classes.empty()) std::cout << " all";
    else for (int cls : target_classes) std::cout << ' ' << cls;
    std::cout << std::endl;
    std::cout << "[Control] Mode=" << aim_mode_name(selected_mode) << " radius=" << tracking_options.aim_radius
              << "px prediction=" << tracking_options.prediction_ms << "ms smoothing=" << hid_smoothing << std::endl;

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    void* ctx = init_yolo26_model(model_path.c_str());
    if (!ctx) {
        std::cerr << "Model Init Failed: " << model_path << std::endl;
        return 1;
    }
    set_yolo26_scores_mode(ctx, scores_mode);
    set_yolo26_nms_iou(ctx, nms_iou);
    if (check_model) {
        cv::Mat input(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
        Detection detections[MAX_DETECTIONS];
        const auto started = std::chrono::steady_clock::now();
        const int count = detect_yolo26(ctx, input.data, CONF_THRES, detections, MAX_DETECTIONS);
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        release_yolo26_model(ctx);
        if (count < 0) {
            std::cerr << "MODEL_CHECK_FAILED: inference failed" << std::endl;
            return 1;
        }
        std::cout << "MODEL_CHECK_PASS model=" << model_path << " detections=" << count
                  << " inference_ms=" << elapsed_ms << std::endl;
        return 0;
    }

    signal(SIGINT, [](int){ running = false; hid_cv.notify_all(); cap_state.cv.notify_all(); });
    signal(SIGTERM, [](int){ running = false; hid_cv.notify_all(); cap_state.cv.notify_all(); });
    signal(SIGPIPE, SIG_IGN);

    // Validate model and camera before creating worker threads.
    v4l2_context_t camera_ctx;
    if (v4l2_open(&camera_ctx, "/dev/video20", 1920, 1080, 180) < 0) {
        release_yolo26_model(ctx);
        return 1;
    }
    if (v4l2_start_stream(&camera_ctx) < 0) {
        std::cerr << "Camera Stream Start Failed" << std::endl;
        v4l2_close(&camera_ctx);
        release_yolo26_model(ctx);
        return 1;
    }
    std::cout << "[System] Manual V4L2 Driver Started" << std::endl;
    
    set_affinity_big(); 
    set_realtime_priority();
    lock_memory();
    if (hid_enabled) hid_init(hid_device.c_str());
    else std::cout << "[HID] Disabled by --no-hid" << std::endl;
    
    std::thread hid_thread;
    if (hid_enabled) hid_thread = std::thread([hid_smoothing](){ set_affinity_little(); hid_worker(hid_smoothing); });
    std::thread web_thread([](){ web_worker(HTTP_PORT, MJPEG_QUALITY, AIM_HEIGHT_RATIO); });
    std::thread cleaner_thread([](){ // Background FD Scrubber
        set_affinity_little();
        while(running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            scrub_sync_files();
        }
    });
    cleaner_thread.detach(); 
    
    std::thread cap_thread(capture_worker, &camera_ctx);
    std::thread web_proc_thread(web_processor_func, &camera_ctx); // Launch processor
    
    cv::Mat letterbox_img(640, 640, CV_8UC3);
    cv::Mat rgb_img(640, 640, CV_8UC3);
    rgb_img.setTo(cv::Scalar(114, 114, 114));
    
    Detection dets[MAX_DETECTIONS];
    auto last_print_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    int exit_status = 0;
    
    const float cx = camera_ctx.width / 2.0f;
    const float cy = camera_ctx.height / 2.0f;

    // 预计算参数
    float r = std::min(640.0f/camera_ctx.height, 640.0f/camera_ctx.width);
    int nw = round(camera_ctx.width*r), nh = round(camera_ctx.height*r);
    int dw = (640-nw)/2, dh = (640-nh)/2;

    while (running) {
        auto t_start = std::chrono::high_resolution_clock::now();
        
        int current_idx = -1;
        {
            std::unique_lock<std::mutex> lock(cap_state.mutex);
            cap_state.cv.wait(lock, [] { return !running || cap_state.new_frame_available; });
            if (!running) break;
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
            if (dma_fd >= 0) src_rga = wrapbuffer_fd(dma_fd, camera_ctx.width, camera_ctx.height, RK_FORMAT_BGR_888);
            else src_rga = wrapbuffer_virtualaddr(raw_data, camera_ctx.width, camera_ctx.height, RK_FORMAT_BGR_888);
            
            void* dst_ptr = rgb_img.data + (dh * 640 + dw) * 3;
            rga_buffer_t dst_rga = wrapbuffer_virtualaddr(dst_ptr, nw, nh, RK_FORMAT_RGB_888); 
            dst_rga.wstride = 640; dst_rga.hstride = nh;
            
            if (imresize(src_rga, dst_rga) != IM_STATUS_SUCCESS) {
                cv::Mat wrapper(camera_ctx.height, camera_ctx.width, CV_8UC3, raw_data, camera_ctx.bytes_per_line);
                preprocess(wrapper, letterbox_img, r, dw, dh);
                cv::cvtColor(letterbox_img, rgb_img, cv::COLOR_BGR2RGB);
            }
        } else {
             cv::Mat encoded(1, camera_ctx.buffers[current_idx].bytes_used, CV_8UC1, raw_data);
             cv::Mat wrapper = cv::imdecode(encoded, cv::IMREAD_COLOR);
             if (wrapper.empty()) { v4l2_manual_qbuf(&camera_ctx, current_idx); continue; }
             preprocess(wrapper, letterbox_img, r, dw, dh);
             cv::cvtColor(letterbox_img, rgb_img, cv::COLOR_BGR2RGB);
        }

        // Inference
        int count = detect_yolo26(ctx, rgb_img.data, CONF_THRES, dets, MAX_DETECTIONS);
        if (count < 0) {
            std::cerr << "NPU inference failed" << std::endl;
            v4l2_manual_qbuf(&camera_ctx, current_idx);
            exit_status = 1;
            running = false;
            break;
        }

        // --- 开始调试代码 ---
        if (frame_count % 10 == 0) { // 每10帧打印一次，避免刷屏太快
            auto now = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count();
            std::cout << "[Debug] Frame: " << frame_count
                      << " | Time: " << ms << "ms"
                      << " | Dets: " << count;
            if (count > 0) {
                 std::cout << " | Best: " << dets[0].score
                           << " Class: " << dets[0].class_id
                           << " Box: [" << (int)dets[0].x1 << "," << (int)dets[0].y1 << ","
                           << (int)dets[0].x2 << "," << (int)dets[0].y2 << "]";
            }
            std::cout << std::endl;
        }
        frame_count++;
        // --- 结束调试代码 ---

        // Associate detections once per track, hold stable selection, and use source pixels.
        const uint64_t control_revision = aim_settings_revision.load();
        const AimTarget target = tracker.update(dets,count,r,dw,dh,camera_ctx.width,camera_ctx.height,
            std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(),
            static_cast<AimMode>(aim_mode.load()));
        if (target.changed || control_revision != last_control_revision) {
            pid_state.tracking_frames = 0;
            std::lock_guard<std::mutex> lock(hid_mutex);
            hid_buffer_x = hid_buffer_y = 0; fire_request = false;
            hid_cv.notify_one();
        }
        last_control_revision = control_revision;
        hid_target_class = target.visible ? target.class_id : -1;
        if (target.visible && hid_enabled && hid_control_enabled) {
            float raw_dx = target.x - cx;
            float raw_dy = target.y - cy;
            
            if (std::abs(raw_dx) < AIM_DEADZONE) raw_dx = 0;
            if (std::abs(raw_dy) < AIM_DEADZONE) raw_dy = 0;

            // A new/reacquired target must not inherit another target's derivative.
            float mx = (PID_KP * raw_dx) + (pid_state.tracking_frames ? PID_KD * (raw_dx - pid_state.last_error_x) : 0);
            float my = (PID_KP * raw_dy) + (pid_state.tracking_frames ? PID_KD * (raw_dy - pid_state.last_error_y) : 0);
            
            pid_state.last_error_x = raw_dx; pid_state.last_error_y = raw_dy;
            pid_state.tracking_frames++;
            
            float predicted_dist = std::sqrt(std::pow(raw_dx * (1.0f - PID_KP), 2) + std::pow(raw_dy * (1.0f - PID_KP), 2));
            const float distance = std::hypot(raw_dx,raw_dy);
            bool safe_to_predict = distance < (AUTO_FIRE_RADIUS * 2.5f);
            bool should_fire = (distance < AUTO_FIRE_RADIUS) || (predicted_dist < AUTO_FIRE_RADIUS && safe_to_predict);

            {
                std::lock_guard<std::mutex> lock(hid_mutex);
                if (hid_control_enabled && aim_settings_revision.load() == control_revision) {
                    fire_request = should_fire;
                    hid_buffer_x += mx * MOUSE_SENSITIVITY;
                    hid_buffer_y += my * MOUSE_SENSITIVITY;
                }
            }
            hid_cv.notify_one();
        } else {
            pid_state.tracking_frames = 0;
            pid_state.last_error_x = 0;
            pid_state.last_error_y = 0;
            {
                std::lock_guard<std::mutex> lock(hid_mutex);
                fire_request = false;
                hid_buffer_x = 0; hid_buffer_y = 0;
            }
            hid_cv.notify_one();
        }

        // 5. Offload Buffer to Web Thread (If available)
        const bool buffer_offloaded = web_submit_frame(&camera_ctx, current_idx, dets, count, r, dw, dh,
                                                      target,tracking_options.aim_radius);
        
        if (!buffer_offloaded) {
             v4l2_manual_qbuf(&camera_ctx, current_idx);
        }

        auto t_end = std::chrono::high_resolution_clock::now();

        if (std::chrono::duration<double>(t_end - last_print_time).count() >= 1.0) {
            std::cout << std::fixed << std::setprecision(1) << "[FPS: " << frame_count << "]\r" << std::flush;
            last_print_time = t_end; frame_count=0;
        }
    }
    
    running = false;
    hid_cv.notify_all();
    cap_state.cv.notify_all();
    if(hid_thread.joinable()) hid_thread.join();
    if(web_thread.joinable()) web_thread.join();
    if(web_proc_thread.joinable()) web_proc_thread.join();
    if(cap_thread.joinable()) cap_thread.join();
    
    release_yolo26_model(ctx);
    v4l2_close(&camera_ctx);
    return exit_status;
}
