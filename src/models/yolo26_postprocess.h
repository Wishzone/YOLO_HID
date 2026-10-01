#ifndef YOLO26_POSTPROCESS_H
#define YOLO26_POSTPROCESS_H

#include "../core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float* data;
    int channels;
    int height;
    int width;
    int is_nhwc;
    int scores_mode; // 0=auto, 1=logits, 2=probabilities
} Yolo26RawHead;

// Raw YOLO26 exports: four LTRB distances followed by class scores.
int yolo26_decode_raw_heads(const Yolo26RawHead* heads, int head_count,
                           int input_width, int input_height, float threshold,
                           Detection* results, int capacity);

// Decode every raw candidate, then keep the best boxes per class. Capacity
// is applied after suppression so early duplicates cannot hide later targets.
int yolo26_decode_raw_heads_nms(const Yolo26RawHead* heads, int head_count,
                               int input_width, int input_height, float threshold,
                               float iou_threshold, Detection* results, int capacity);

// In-place, score-ordered NMS. Overlapping boxes of different classes survive.
int yolo26_nms(Detection* detections, int count, float iou_threshold, int capacity);

// End-to-end exports: XYXY coordinates, probability, and class index.
int yolo26_decode_detections(const float* data, int rows, int box_last,
                            int input_width, int input_height, float threshold,
                            Detection* results, int capacity);

#ifdef __cplusplus
}
#endif
#endif
