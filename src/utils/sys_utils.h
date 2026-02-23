#ifndef SYS_UTILS_H
#define SYS_UTILS_H

#include <opencv2/opencv.hpp>

void set_realtime_priority();
void lock_memory();
void set_affinity_big();
void set_affinity_little();
void preprocess(const cv::Mat& img, cv::Mat& out, float& r, int& dw, int& dh);

#endif // SYS_UTILS_H
