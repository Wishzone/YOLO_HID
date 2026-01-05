#ifndef YOLO_RKNN_H
#define YOLO_RKNN_H

#include <rknn_api.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float x1, y1, x2, y2;
    float score;
    int class_id;
} Detection;

void* init_model(const char* model_path);
int detect(void* ctx_ptr, unsigned char* input_data, float conf_thres, float nms_thres, Detection* results, int max_results);
void release_model(void* ctx_ptr);

#ifdef __cplusplus
}
#endif

#endif
