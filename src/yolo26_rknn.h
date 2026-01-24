#ifndef YOLO26_RKNN_H
#define YOLO26_RKNN_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float x1, y1, x2, y2;
    float score;
    int class_id;
} Detection26;

void* init_yolo26_model(const char* model_path);
int detect_yolo26(void* ctx_ptr, unsigned char* input_data, float conf_thres, float nms_thres, Detection26* results, int max_results);
void release_yolo26_model(void* ctx_ptr);

#ifdef __cplusplus
}
#endif

#endif
