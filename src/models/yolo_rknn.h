#ifndef YOLO_RKNN_H
#define YOLO_RKNN_H

#include <rknn_api.h>
#include <stdbool.h>
#include "../core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

void* init_model(const char* model_path);
int detect(void* ctx_ptr, unsigned char* input_data, float conf_thres, float nms_thres, Detection* results, int max_results);
void release_model(void* ctx_ptr);

#ifdef __cplusplus
}
#endif

#endif
