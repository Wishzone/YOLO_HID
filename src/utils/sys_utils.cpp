#include "sys_utils.h"
#include <iostream>
#include <sched.h>
#include <sys/mman.h>

void set_realtime_priority() {
    struct sched_param param;
    param.sched_priority = 50; 
    if (sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
        std::cerr << "[System] RT Priority Failed (Run as Root)" << std::endl;
    } else {
        std::cout << "[System] Real-Time Priority Enabled" << std::endl;
    }
}

void lock_memory() {
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        std::cerr << "[System] Memory Lock Failed" << std::endl;
    }
}

void set_affinity_big() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(4, &mask); CPU_SET(5, &mask); CPU_SET(6, &mask); CPU_SET(7, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
}

void set_affinity_little() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask); CPU_SET(1, &mask); CPU_SET(2, &mask); CPU_SET(3, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
}

void preprocess(const cv::Mat& img, cv::Mat& out, float& r, int& dw, int& dh) {
    float h = img.rows, w = img.cols;
    r = std::min(640.0f/h, 640.0f/w);
    int nw = round(w*r), nh = round(h*r);
    dw = (640-nw)/2; dh = (640-nh)/2;
    
    static cv::Mat resized;
    if (resized.empty() || resized.rows != nh || resized.cols != nw) resized.create(nh, nw, img.type());
    
    cv::resize(img, resized, cv::Size(nw, nh));
    out.setTo(cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(dw, dh, nw, nh)));
}
