#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <dirent.h>
#include "v4l2_driver.h"

#define PERROR(msg) do { fprintf(stderr, "[V4L2] " msg ": %s\n", strerror(errno)); } while(0)

int v4l2_open(v4l2_context_t* ctx, const char* dev_path, int width, int height, int fps) {
    memset(ctx, 0, sizeof(v4l2_context_t));
    
    ctx->fd = open(dev_path, O_RDWR | O_NONBLOCK, 0);
    if (ctx->fd == -1) {
        PERROR("Cannot open device");
        return -1;
    }

    struct v4l2_capability cap;
    if (ioctl(ctx->fd, VIDIOC_QUERYCAP, &cap) == -1) {
        PERROR("VIDIOC_QUERYCAP failed");
        close(ctx->fd);
        return -1;
    }

    if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE) {
        ctx->type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        ctx->is_mplane = 1;
    } else if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) {
        ctx->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ctx->is_mplane = 0;
    } else {
        close(ctx->fd);
        return -1;
    }

    // Set Format (BGR24 preferred, MJPEG fallback)
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = ctx->type;
    
    if (ctx->is_mplane) {
        fmt.fmt.pix_mp.width = width;
        fmt.fmt.pix_mp.height = height;
        fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_BGR24;
        fmt.fmt.pix_mp.field = V4L2_FIELD_ANY;
        fmt.fmt.pix_mp.num_planes = 1;
    } else {
        fmt.fmt.pix.width = width;
        fmt.fmt.pix.height = height;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_BGR24;
        fmt.fmt.pix.field = V4L2_FIELD_ANY;
    }

    if (ioctl(ctx->fd, VIDIOC_S_FMT, &fmt) == -1) {
        // Fallback MJPEG
        if (ctx->is_mplane) fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_MJPEG;
        else fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        
        if (ioctl(ctx->fd, VIDIOC_S_FMT, &fmt) == -1) {
            PERROR("VIDIOC_S_FMT failed");
            close(ctx->fd);
            return -1;
        }
    }
    
    // Update ctx width/height/format
    if (ctx->is_mplane) {
        ctx->width = fmt.fmt.pix_mp.width;
        ctx->height = fmt.fmt.pix_mp.height;
        ctx->format = fmt.fmt.pix_mp.pixelformat;
    } else {
        ctx->width = fmt.fmt.pix.width;
        ctx->height = fmt.fmt.pix.height;
        ctx->format = fmt.fmt.pix.pixelformat;
    }
    
    fprintf(stderr, "[V4L2] Fmt: %c%c%c%c %dx%d\n",
        (ctx->format) & 0xFF, (ctx->format>>8) & 0xFF, (ctx->format>>16) & 0xFF, (ctx->format>>24) & 0xFF,
        ctx->width, ctx->height);

    // Set FPS
    struct v4l2_streamparm streamparm;
    memset(&streamparm, 0, sizeof(streamparm));
    streamparm.type = ctx->type;
    streamparm.parm.capture.timeperframe.numerator = 1;
    streamparm.parm.capture.timeperframe.denominator = fps;
    ioctl(ctx->fd, VIDIOC_S_PARM, &streamparm);
    
    // Request Buffers (Quad Buffering for Pipeline)
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 4; // <--- 4 Buffers
    req.type = ctx->type;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(ctx->fd, VIDIOC_REQBUFS, &req) == -1) {
        PERROR("VIDIOC_REQBUFS failed");
        close(ctx->fd);
        return -1;
    }

    ctx->buffers = (v4l2_buffer_mapping_t*)calloc(req.count, sizeof(v4l2_buffer_mapping_t));
    ctx->n_buffers = req.count;

    for (unsigned int i = 0; i < req.count; ++i) {
        struct v4l2_buffer buf;
        struct v4l2_plane planes[8];
        memset(&buf, 0, sizeof(buf));
        memset(planes, 0, sizeof(planes));
        
        buf.type = ctx->type;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        
        if (ctx->is_mplane) {
            buf.m.planes = planes;
            buf.length = 8;
        }

        if (ioctl(ctx->fd, VIDIOC_QUERYBUF, &buf) == -1) {
            close(ctx->fd);
            return -1;
        }

        if (ctx->is_mplane) {
            ctx->buffers[i].length = buf.m.planes[0].length;
            ctx->buffers[i].start = mmap(NULL, buf.m.planes[0].length,
                  PROT_READ | PROT_WRITE, MAP_SHARED,
                  ctx->fd, buf.m.planes[0].m.mem_offset);
        } else {
            ctx->buffers[i].length = buf.length;
            ctx->buffers[i].start = mmap(NULL, buf.length,
                  PROT_READ | PROT_WRITE, MAP_SHARED,
                  ctx->fd, buf.m.offset);
        }

        if (ctx->buffers[i].start == MAP_FAILED) {
            close(ctx->fd);
            return -1;
        }
        
        // Export DMABUF
        struct v4l2_exportbuffer expbuf;
        memset(&expbuf, 0, sizeof(expbuf));
        expbuf.type = ctx->type;
        expbuf.index = i;
        expbuf.plane = 0; 
        expbuf.flags = O_CLOEXEC | O_RDWR;
        
        if (ioctl(ctx->fd, VIDIOC_EXPBUF, &expbuf) == -1) {
            ctx->buffers[i].dma_fd = -1;
        } else {
            ctx->buffers[i].dma_fd = expbuf.fd;
        }
    }

    return 0;
}

int v4l2_start_stream(v4l2_context_t* ctx) {
    for (unsigned int i = 0; i < ctx->n_buffers; ++i) {
        struct v4l2_buffer buf;
        struct v4l2_plane planes[8];
        memset(&buf, 0, sizeof(buf));
        memset(planes, 0, sizeof(planes));
        
        buf.type = ctx->type;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        
        if (ctx->is_mplane) {
            buf.m.planes = planes;
            buf.length = 1;
        }
        
        if (ioctl(ctx->fd, VIDIOC_QBUF, &buf) == -1) return -1;
    }

    enum v4l2_buf_type type = (enum v4l2_buf_type)ctx->type;
    return ioctl(ctx->fd, VIDIOC_STREAMON, &type);
}

void v4l2_stop_stream(v4l2_context_t* ctx) {
    enum v4l2_buf_type type = (enum v4l2_buf_type)ctx->type;
    ioctl(ctx->fd, VIDIOC_STREAMOFF, &type);
}

void v4l2_close(v4l2_context_t* ctx) {
    v4l2_stop_stream(ctx);
    for (unsigned int i = 0; i < ctx->n_buffers; ++i) {
        if (ctx->buffers[i].start)
            munmap(ctx->buffers[i].start, ctx->buffers[i].length);
    }
    free(ctx->buffers);
    close(ctx->fd);
}

int v4l2_manual_dqbuf(v4l2_context_t* ctx, struct v4l2_buffer* buf) {
    memset(buf, 0, sizeof(*buf));
    // Caller must provide buf->m.planes if MPLANE
    
    struct v4l2_plane planes[8]; // Default local planes if caller relies on v4l2 defaults? 
    // No, standard `buf` struct doesn't hold planes, it holds pointer.
    // We assume caller has set buf->m.planes = some_array if MPLANE.
    // But to be safe properly mimicking libc:
    // If ctx->is_mplane is true, buf->m.planes must point to valid memory.
    // The Main Loop in C++ provides this.
    
    // However, if we just want to be robust:
    buf->type = ctx->type;
    buf->memory = V4L2_MEMORY_MMAP;
    if(ctx->is_mplane && buf->m.planes == NULL) {
         buf->m.planes = planes;
         buf->length = 8;
    }

    if (ioctl(ctx->fd, VIDIOC_DQBUF, buf) == -1) return -1;
    return 0;
}

int v4l2_manual_qbuf(v4l2_context_t* ctx, int index) {
    struct v4l2_buffer buf;
    struct v4l2_plane planes[8];
    memset(&buf, 0, sizeof(buf));
    memset(planes, 0, sizeof(planes));
    
    buf.type = ctx->type;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = index;
    
    if (ctx->is_mplane) {
        buf.m.planes = planes;
        buf.length = 1;
    }
    
    if (ioctl(ctx->fd, VIDIOC_QBUF, &buf) == -1) return -1;
    return 0;
}

void scrub_sync_files() {
    DIR *dp = opendir("/proc/self/fd");
    if (!dp) return;
    
    struct dirent *ep;
    char path[64];
    char target[64];
    
    while ((ep = readdir(dp))) {
        if (ep->d_name[0] == '.') continue;
        int fd = atoi(ep->d_name);
        if (fd < 100) continue; 
        
        snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
        ssize_t len = readlink(path, target, sizeof(target)-1);
        if (len > 0) {
            target[len] = '\0';
            if (strstr(target, "sync_file") != NULL) {
                close(fd);
            }
        }
    }
    closedir(dp);
}
