#ifndef V4L2_DRIVER_H
#define V4L2_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <linux/videodev2.h>

typedef struct {
    void* start;
    size_t length;
    int dma_fd;
} v4l2_buffer_mapping_t;

typedef struct {
    int fd;
    uint32_t width;
    uint32_t height;
    uint32_t format; 
    v4l2_buffer_mapping_t* buffers;
    unsigned int n_buffers;
    
    // Internal state
    uint32_t type; 
    int is_mplane;
} v4l2_context_t;

// Lifecycle
int v4l2_open(v4l2_context_t* ctx, const char* dev_path, int width, int height, int fps);
int v4l2_start_stream(v4l2_context_t* ctx);
void v4l2_stop_stream(v4l2_context_t* ctx);
void v4l2_close(v4l2_context_t* ctx);

// Manual Buffer Control (Async / Zero-Copy)
int v4l2_manual_dqbuf(v4l2_context_t* ctx, struct v4l2_buffer* buf);
int v4l2_manual_qbuf(v4l2_context_t* ctx, int index);

// Utilities
void scrub_sync_files();

#ifdef __cplusplus
}
#endif

#endif
