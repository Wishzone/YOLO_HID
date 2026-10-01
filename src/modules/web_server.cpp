#include "web_server.h"
#include "hid_mouse.h"
#include "../core/globals.h"
#include "../utils/sys_utils.h"
#include <rga/im2d.h>
#include <rga/RgaApi.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <sys/poll.h>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cerrno>

WebData web_data_buffer;
std::mutex web_mutex;

struct WebTransfer {
    int index = -1;
    bool busy = false;
    std::vector<Detection> detections;
    float r = 1;
    int dw = 0, dh = 0;
};
static WebTransfer transfer;
static std::mutex transfer_mutex;
static std::condition_variable transfer_cv;

bool web_submit_frame(v4l2_context_t* ctx, int index, const Detection* dets,
                      int count, float r, int dw, int dh) {
    if (active_connections <= 0 || index < 0 ||
        static_cast<unsigned>(index) >= ctx->n_buffers) return false;
    std::lock_guard<std::mutex> lock(transfer_mutex);
    if (transfer.busy) return false;
    transfer.busy = true;
    transfer.index = index;
    transfer.detections.assign(dets, dets + count);
    transfer.r = r; transfer.dw = dw; transfer.dh = dh;
    transfer_cv.notify_one();
    return true;
}

void web_processor_func(v4l2_context_t* ctx) {
    set_affinity_little();
    for (;;) {
        WebTransfer frame;
        {
            std::unique_lock<std::mutex> lock(transfer_mutex);
            transfer_cv.wait_for(lock, std::chrono::milliseconds(20), [] {
                return transfer.index >= 0 || !running;
            });
            if (transfer.index < 0) { if (!running) break; else continue; }
            frame = transfer;
            transfer.index = -1;
            // Keep busy until the original V4L2 frame has been copied and returned.
        }
        cv::Mat image;
        try {
            const auto& buffer = ctx->buffers[frame.index];
            if (ctx->format == V4L2_PIX_FMT_BGR24) {
                image.create(ctx->height, ctx->width, CV_8UC3);
                IM_STATUS copied = IM_STATUS_FAILED;
                if (buffer.dma_fd >= 0 && ctx->bytes_per_line == ctx->width * 3) {
                    auto src = wrapbuffer_fd(buffer.dma_fd, ctx->width, ctx->height, RK_FORMAT_BGR_888);
                    auto dst = wrapbuffer_virtualaddr(image.data, ctx->width, ctx->height, RK_FORMAT_BGR_888);
                    copied = imcopy(src, dst);
                }
                if (copied != IM_STATUS_SUCCESS) {
                    cv::Mat raw(ctx->height, ctx->width, CV_8UC3, buffer.start, ctx->bytes_per_line);
                    raw.copyTo(image);
                }
            } else if (buffer.bytes_used > 0 && buffer.bytes_used <= buffer.length) {
                cv::Mat encoded(1, buffer.bytes_used, CV_8UC1, buffer.start);
                image = cv::imdecode(encoded, cv::IMREAD_COLOR);
            }
        } catch (const cv::Exception& error) {
            std::cerr << "[Web] Frame copy failed: " << error.what() << std::endl;
        }
        v4l2_manual_qbuf(ctx, frame.index);
        if (!image.empty()) {
            std::lock_guard<std::mutex> lock(web_mutex);
            web_data_buffer.img = image;
            web_data_buffer.dets = std::move(frame.detections);
            web_data_buffer.r = frame.r;
            web_data_buffer.dw = frame.dw;
            web_data_buffer.dh = frame.dh;
            web_data_buffer.valid = true;
            ++web_data_buffer.frame_id;
        }
        {
            std::lock_guard<std::mutex> lock(transfer_mutex);
            transfer.busy = false;
        }
    }
}

int draw_detection_overlay(cv::Mat& image, const std::vector<Detection>& detections,
                           float r, int dw, int dh, float aim_height_ratio) {
    if (image.empty() || !std::isfinite(r) || r <= 0) return 0;
    const cv::Scalar colors[] = {{0,255,0}, {0,210,255}, {255,160,0}, {255,0,220}};
    int drawn = 0;
    for (const auto& det : detections) {
        if (!std::isfinite(det.x1) || !std::isfinite(det.y1) ||
            !std::isfinite(det.x2) || !std::isfinite(det.y2)) continue;
        const int x1 = cvRound(std::max(0.0f, std::min(float(image.cols - 1), (det.x1 - dw) / r)));
        const int y1 = cvRound(std::max(0.0f, std::min(float(image.rows - 1), (det.y1 - dh) / r)));
        const int x2 = cvRound(std::max(0.0f, std::min(float(image.cols - 1), (det.x2 - dw) / r)));
        const int y2 = cvRound(std::max(0.0f, std::min(float(image.rows - 1), (det.y2 - dh) / r)));
        if (x2 <= x1 || y2 <= y1) continue;
        const cv::Scalar color = colors[static_cast<unsigned>(det.class_id) % 4];
        cv::rectangle(image, {x1,y1}, {x2,y2}, color, 3);
        std::ostringstream label;
        label << "class " << det.class_id << "  " << std::fixed << std::setprecision(0) << det.score*100 << '%';
        const cv::Point text(x1, std::max(22, y1 - 8));
        cv::putText(image, label.str(), text, cv::FONT_HERSHEY_SIMPLEX, 0.65, {0,0,0}, 4);
        cv::putText(image, label.str(), text, cv::FONT_HERSHEY_SIMPLEX, 0.65, color, 2);
        cv::circle(image, {(x1+x2)/2, cvRound(y1+(y2-y1)*aim_height_ratio)}, 4, {0,0,255}, -1);
        ++drawn;
    }
    return drawn;
}

static bool send_all(int fd, const void* data, size_t size) {
    const char* bytes = static_cast<const char*>(data);
    while (size > 0) {
        const ssize_t sent = send(fd, bytes, size, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) return false;
        bytes += sent; size -= sent;
    }
    return true;
}

static void respond(int fd, const char* content_type, const std::string& body) {
    const std::string header = std::string("HTTP/1.1 200 OK\r\nContent-Type: ") + content_type +
        "\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n";
    if (send_all(fd, header.data(), header.size())) send_all(fd, body.data(), body.size());
}

static void handle_client(int socket, int quality, float aim_height_ratio) {
    const timeval timeout{3,0};
    setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    char request[2048] = {};
    const ssize_t length = recv(socket, request, sizeof(request)-1, 0);
    if (length <= 0) { close(socket); --active_connections; return; }
    const std::string first_line(request, static_cast<size_t>(length));
    if (first_line.find("POST /hid/pause ") == 0 || first_line.find("POST /hid/resume ") == 0) {
        {
            std::lock_guard<std::mutex> lock(hid_mutex);
            hid_control_enabled = first_line.find("POST /hid/resume ") == 0;
            hid_buffer_x = hid_buffer_y = 0;
            fire_request = false;
        }
        hid_cv.notify_all();
        respond(socket, "application/json", "{\"ok\":true}");
    } else if (first_line.find("GET /status ") == 0) {
        const auto hid = hid_get_status();
        std::ostringstream json;
        json << "{\"usb_state\":\"" << hid_usb_state() << "\",\"hid_open\":" << (hid.opened ? "true" : "false")
             << ",\"hid_reports\":" << hid.reports << ",\"hid_errors\":" << hid.errors
             << ",\"target_class\":" << hid_target_class.load()
             << ",\"hid_enabled\":" << (hid_control_enabled.load() ? "true" : "false");
        {
            std::lock_guard<std::mutex> lock(web_mutex);
            json << ",\"frame_id\":" << web_data_buffer.frame_id
                 << ",\"detections\":" << web_data_buffer.dets.size() << ",\"classes\":[";
            for (size_t i=0; i<web_data_buffer.dets.size(); ++i) {
                if (i) json << ',';
                json << web_data_buffer.dets[i].class_id;
            }
        }
        json << "]}";
        respond(socket, "application/json", json.str());
    } else if (first_line.find("GET /stream") == 0 || first_line.find("GET /snapshot.jpg") == 0) {
        const bool snapshot = first_line.find("GET /snapshot.jpg") == 0;
        if (!snapshot) {
            const std::string header = "HTTP/1.1 200 OK\r\nCache-Control: no-store\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
            if (!send_all(socket, header.data(), header.size())) { close(socket); --active_connections; return; }
        }
        uint64_t last_frame_id = 0;
        const auto started = std::chrono::steady_clock::now();
        while (running) {
            char byte;
            if (recv(socket, &byte, 1, MSG_DONTWAIT | MSG_PEEK) == 0) break;
            cv::Mat image;
            std::vector<Detection> detections;
            float r = 1; int dw = 0, dh = 0;
            {
                std::lock_guard<std::mutex> lock(web_mutex);
                if (web_data_buffer.valid && web_data_buffer.frame_id > last_frame_id) {
                    image = web_data_buffer.img.clone();
                    detections = web_data_buffer.dets;
                    r = web_data_buffer.r; dw = web_data_buffer.dw; dh = web_data_buffer.dh;
                    last_frame_id = web_data_buffer.frame_id;
                }
            }
            if (image.empty()) {
                if (snapshot && std::chrono::steady_clock::now()-started > std::chrono::seconds(3)) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            draw_detection_overlay(image, detections, r, dw, dh, aim_height_ratio);
            std::vector<uchar> jpeg;
            if (!cv::imencode(".jpg", image, jpeg, {cv::IMWRITE_JPEG_QUALITY, quality})) continue;
            if (snapshot) {
                respond(socket, "image/jpeg", std::string(reinterpret_cast<char*>(jpeg.data()), jpeg.size()));
                break;
            }
            const std::string header = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + std::to_string(jpeg.size()) + "\r\n\r\n";
            if (!send_all(socket, header.data(), header.size()) || !send_all(socket, jpeg.data(), jpeg.size()) ||
                !send_all(socket, "\r\n", 2)) break;
        }
    } else {
        respond(socket, "text/html; charset=utf-8", R"HTML(<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO 检测预览</title>
<style>body{margin:0;background:#111;color:#eee;font:14px sans-serif}header{padding:12px;background:#222;display:flex;gap:16px;align-items:center;flex-wrap:wrap}button{padding:8px 16px;cursor:pointer}main{display:flex;justify-content:center}img{max-width:100%;max-height:calc(100vh - 90px)}</style>
<header><button id="hid" disabled>读取 HID 状态</button><span id="status">正在连接…</span></header>
<main><img src="/stream" alt="HDMI 检测画面"></main>
<script>
let enabled=false;
async function update(){
 try{
  const s=await(await fetch('/status',{cache:'no-store'})).json(); enabled=s.hid_enabled;
  const c=[...new Set(s.classes)].join(', ')||'无';
  const button=document.getElementById('hid'); button.disabled=!s.hid_open;
  button.textContent=!s.hid_open?'HID 未打开':enabled?'暂停 HID 鼠标':'启用 HID 鼠标';
  document.getElementById('status').textContent=`检测框 ${s.detections} | 类别 ${c} | 控制目标 ${s.target_class<0?'未选中':'类别 '+s.target_class} | USB ${s.usb_state} | HID ${enabled?'运行':'暂停'} | 已发送 ${s.hid_reports} | 写入失败 ${s.hid_errors}`;
 }catch(e){document.getElementById('status').textContent='状态连接中断，请刷新页面';}
}
document.getElementById('hid').onclick=async()=>{await fetch(enabled?'/hid/pause':'/hid/resume',{method:'POST'});await update();};
update();setInterval(update,1000);
</script></html>)HTML");
    }
    close(socket);
    --active_connections;
}

void web_worker(int port, int quality, float aim_height_ratio) {
    set_affinity_little();
    const int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) { std::cerr << "[Web] socket failed" << std::endl; return; }
    const int reuse = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{};
    address.sin_family = AF_INET; address.sin_addr.s_addr = INADDR_ANY; address.sin_port = htons(port);
    if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 || listen(fd, 5) < 0) {
        std::cerr << "[Web] Cannot listen on port " << port << std::endl; close(fd); return;
    }
    while (running) {
        pollfd pfd{fd, POLLIN, 0};
        if (poll(&pfd, 1, 100) > 0) {
            const int client = accept4(fd, nullptr, nullptr, SOCK_CLOEXEC);
            if (client >= 0) {
                if (active_connections >= 5) close(client);
                else {
                    ++active_connections;
                    std::thread(handle_client, client, quality, aim_height_ratio).detach();
                }
            }
        }
    }
    close(fd);
    // Detached clients must finish before the global image and mutex are destroyed.
    while (active_connections > 0) std::this_thread::sleep_for(std::chrono::milliseconds(10));
}
