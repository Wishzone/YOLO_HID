#ifndef CAMERA_CAPTURE_H
#define CAMERA_CAPTURE_H

#include "v4l2_driver.h"
#include <mutex>
#include <condition_variable>

struct CapState {
    std::mutex mutex;
    std::condition_variable cv;
    int latest_idx = -1;
    bool new_frame_available = false;
    v4l2_context_t* ctx = nullptr;
};

extern CapState cap_state;

void capture_worker(v4l2_context_t* ctx);

#endif // CAMERA_CAPTURE_H
