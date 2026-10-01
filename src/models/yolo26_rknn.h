#ifndef YOLO26_RKNN_H
#define YOLO26_RKNN_H

#include "../core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

void* init_yolo26_model(const char* model_path);
// 0=auto, 1=class logits, 2=class probabilities. Applies to raw heads only.
void set_yolo26_scores_mode(void* ctx_ptr, int mode);
// Class-aware NMS for raw heads only. Default IoU is 0.5; end-to-end output is unchanged.
void set_yolo26_nms_iou(void* ctx_ptr, float iou);
// Returns the detection count, or -1 if the RKNN inference fails.
int detect_yolo26(void* ctx_ptr, unsigned char* input_data, float conf_thres, Detection* results, int max_results);
void release_yolo26_model(void* ctx_ptr);

#ifdef __cplusplus
}
#endif

#endif
