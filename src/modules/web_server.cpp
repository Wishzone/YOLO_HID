#include "web_server.h"
#include "../core/globals.h"
#include "../utils/sys_utils.h"
#include <rga/im2d.h>
#include <rga/RgaApi.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <thread>
#include <sys/poll.h>

WebData web_data_buffer;
std::mutex web_mutex;

std::atomic<int> web_transfer_buf_idx{-1};
std::mutex web_transfer_mutex;
std::vector<Detection> web_transfer_dets;
float web_transfer_r; 
int web_transfer_dw, web_transfer_dh;
int web_transfer_src_dma;
int web_transfer_src_w, web_transfer_src_h;
int web_transfer_length;
void* web_transfer_data = nullptr;

void web_processor_func(v4l2_context_t* v4l2_ctx) {
    set_affinity_little(); 
    
    while(running) {
        int idx = web_transfer_buf_idx.exchange(-1);
        if (idx == -1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        
        std::vector<Detection> local_dets;
        {
            std::lock_guard<std::mutex> lock(web_transfer_mutex);
            local_dets = web_transfer_dets;
        }
        
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
             cv::Mat raw_mat(1, web_transfer_length, CV_8UC1, web_transfer_data);
             cv::Mat decoded;
             cv::imdecode(raw_mat, cv::IMREAD_COLOR, &decoded);
             if(!decoded.empty()) decoded.copyTo(web_data_buffer.img);
        }
        
        web_data_buffer.dets = local_dets;
        web_data_buffer.r = web_transfer_r;
        web_data_buffer.dw = web_transfer_dw;
        web_data_buffer.dh = web_transfer_dh;
        web_data_buffer.valid = true;
        web_data_buffer.frame_id++;
        
        lock.unlock(); 
        
        v4l2_manual_qbuf(v4l2_ctx, idx);
    }
}

static void handle_client(int client_socket, int mjpeg_quality, float aim_height_ratio) {
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
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, mjpeg_quality};
        cv::Mat local_frame;
        std::vector<Detection> local_dets;
        float r; int dw, dh;
        uint64_t last_frame_id = 0;
        
        while (running) {
            char c;
            if (recv(client_socket, &c, 1, MSG_DONTWAIT | MSG_PEEK) == 0) break;
            
            bool has_new = false;
            {
                std::unique_lock<std::mutex> lock(web_mutex, std::try_to_lock);
                if (lock.owns_lock() && web_data_buffer.valid) {
                    if (web_data_buffer.frame_id > last_frame_id) {
                        web_data_buffer.img.copyTo(local_frame);
                        local_dets = web_data_buffer.dets;
                        r = web_data_buffer.r; dw = web_data_buffer.dw; dh = web_data_buffer.dh;
                        last_frame_id = web_data_buffer.frame_id;
                        has_new = true;
                    }
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
                    float cy = y1 + (y2 - y1) * aim_height_ratio;
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

void web_worker(int port, int mjpeg_quality, float aim_height_ratio) {
    set_affinity_little();
    int server_fd;
    struct sockaddr_in address;
    int opt = 1;

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);
    
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) return;
    listen(server_fd, 5);

    while (running) {
        struct pollfd pfd = {server_fd, POLLIN, 0};
        if (poll(&pfd, 1, 500) > 0) {
            int addrlen = sizeof(address);
            int new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
            if (new_socket >= 0) {
                if (active_connections >= 5) close(new_socket);
                else std::thread(handle_client, new_socket, mjpeg_quality, aim_height_ratio).detach();
            }
        }
    }
    close(server_fd);
}
