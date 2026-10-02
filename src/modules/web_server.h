#ifndef WEB_SERVER_H
#define WEB_SERVER_H

#include "v4l2_driver.h"
#include "../core/types.h"
#include "target_tracker.h"
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
    AimTarget target;
    float aim_radius = 0;
};

extern WebData web_data_buffer;
extern std::mutex web_mutex;

bool web_submit_frame(v4l2_context_t* ctx, int index, const Detection* dets,
                      int count, float r, int dw, int dh,
                      const AimTarget& target = AimTarget(), float aim_radius = 0);
int draw_detection_overlay(cv::Mat& frame, const std::vector<Detection>& detections,
                           float r, int dw, int dh, float aim_height_ratio);
void draw_target_overlay(cv::Mat& frame, const AimTarget& target, float aim_radius);
void web_processor_func(v4l2_context_t* v4l2_ctx);
void web_worker(int port, int mjpeg_quality, float aim_height_ratio);

#endif // WEB_SERVER_H
