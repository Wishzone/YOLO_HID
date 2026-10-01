#include "../src/modules/web_server.h"
#include "../src/core/globals.h"
#include <cassert>
#include <limits>
#include <thread>
#include <chrono>
#include <cstdio>

static std::atomic<bool> returning_buffer(false);

extern "C" int __wrap_v4l2_manual_qbuf(v4l2_context_t*, int index) {
    assert(index == 0);
    returning_buffer = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return 0;
}

int main() {
    cv::Mat image(1080,1920,CV_8UC3,cv::Scalar(0,0,0));
    // Nonzero classes must be displayed, and letterbox padding must be removed.
    const std::vector<Detection> detections = {{100,200,150,300,0.9f,2}, {200,220,220,240,0.8f,3}};
    assert(draw_detection_overlay(image,detections,1.0f/3,0,140,0.2f) == 2);
    assert(image.at<cv::Vec3b>(180,300) != cv::Vec3b(0,0,0));
    assert(image.at<cv::Vec3b>(240,600) != cv::Vec3b(0,0,0));
    assert(draw_detection_overlay(image,{{0,0,0,0,1,0}},1,0,0,0.2f) == 0);
    assert(draw_detection_overlay(image,detections,0,0,0,0.2f) == 0);
    cv::Mat raw(64,64,CV_8UC3,cv::Scalar(20,40,60));
    v4l2_buffer_mapping_t buffer{};
    buffer.start = raw.data; buffer.length = raw.total()*3; buffer.bytes_used = buffer.length; buffer.dma_fd = -1;
    v4l2_context_t ctx{};
    ctx.width = 64; ctx.height = 64; ctx.format = V4L2_PIX_FMT_BGR24;
    ctx.bytes_per_line = 64*3; ctx.buffers = &buffer; ctx.n_buffers = 1;
    active_connections = 1;
    assert(web_submit_frame(&ctx,0,detections.data(),detections.size(),1,0,0));
    std::thread processor([&] { web_processor_func(&ctx); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!returning_buffer && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    assert(returning_buffer);
    assert(!web_submit_frame(&ctx,0,detections.data(),detections.size(),2,10,10));
    for (;;) {
        { std::lock_guard<std::mutex> lock(web_mutex); if (web_data_buffer.valid) break; }
        assert(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    {
        std::lock_guard<std::mutex> lock(web_mutex);
        assert(web_data_buffer.valid && web_data_buffer.dets[0].class_id == 2);
        assert(web_data_buffer.r == 1 && web_data_buffer.dw == 0);
        assert(web_data_buffer.img.at<cv::Vec3b>(0,0) == cv::Vec3b(20,40,60));
    }
    running = false;
    processor.join();
    puts("PASS Web: all-class overlays, coordinates, CPU copy and frame ownership");
}
