#ifndef WEB_SERVER_H
#define WEB_SERVER_H

#include "v4l2_driver.h"
#include "../core/types.h"
#include <vector>
#include <mutex>
#include <atomic>
#include <opencv2/opencv.hpp>

struct WebData {
    cv::Mat img;
    std::vector<Detection> dets;
    float r; int dw, dh;
    bool valid = false;
    uint64_t frame_id = 0;
};

extern WebData web_data_buffer;
extern std::mutex web_mutex;

extern std::atomic<int> web_transfer_buf_idx;
extern std::mutex web_transfer_mutex;
extern std::vector<Detection> web_transfer_dets;
extern float web_transfer_r; 
extern int web_transfer_dw, web_transfer_dh;
extern int web_transfer_src_dma;
extern int web_transfer_src_w, web_transfer_src_h;
extern int web_transfer_length;
extern void* web_transfer_data;

void web_processor_func(v4l2_context_t* v4l2_ctx);
void web_worker(int port, int mjpeg_quality, float aim_height_ratio);

#endif // WEB_SERVER_H
