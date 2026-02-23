#include "camera_capture.h"
#include "../core/globals.h"
#include <sys/select.h>
#include <cstring>

CapState cap_state;

void capture_worker(v4l2_context_t* ctx) {
    cap_state.ctx = ctx;
    
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
