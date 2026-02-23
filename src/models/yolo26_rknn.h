#ifndef YOLO26_RKNN_H
#define YOLO26_RKNN_H

#include "../core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

void* init_yolo26_model(const char* model_path);
int detect_yolo26(void* ctx_ptr, unsigned char* input_data, float conf_thres, float nms_thres, Detection* results, int max_results);
void release_yolo26_model(void* ctx_ptr);

#ifdef __cplusplus
}
#endif

#endif
