#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <rknn_api.h>
#include "yolo26_rknn.h"

#define MAX_DETECTIONS 100

// Standard YOLOv5 Anchors (640x640)
// Stride 8, 16, 32
static const float anchors[3][6] = {
    {10.0, 13.0, 16.0, 30.0, 33.0, 23.0},   // P3
    {30.0, 61.0, 62.0, 45.0, 59.0, 119.0},  // P4
    {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} // P5
};

typedef struct {
    rknn_context ctx;
    bool is_init;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_width;
    int model_height;
} RKNN_Context;

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// IOU for NMS
static float iou(Detection26* a, Detection26* b) {
    float xx1 = fmaxf(a->x1, b->x1);
    float yy1 = fmaxf(a->y1, b->y1);
    float xx2 = fminf(a->x2, b->x2);
    float yy2 = fminf(a->y2, b->y2);

    float w = fmaxf(0.0f, xx2 - xx1);
    float h = fmaxf(0.0f, yy2 - yy1);
    float inter = w * h;

    float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);

    return inter / (area_a + area_b - inter);
}

static int compare_dets(const void* a, const void* b) {
    Detection26* det_a = (Detection26*)a;
    Detection26* det_b = (Detection26*)b;
    // Sort descending
    if (det_a->score > det_b->score) return -1;
    if (det_a->score < det_b->score) return 1;
    return 0;
}

static void nms_process(Detection26* dets, int* count, float threshold) {
    int keep[MAX_DETECTIONS];
    int keep_count = 0;
    int suppressed[MAX_DETECTIONS] = {0};

    // Sort by score descending (Optimized with qsort)
    qsort(dets, *count, sizeof(Detection26), compare_dets);

    for (int i = 0; i < *count; i++) {
        if (suppressed[i]) continue;
        keep[keep_count++] = i;
        for (int j = i + 1; j < *count; j++) {
            if (suppressed[j]) continue;
            if (iou(&dets[i], &dets[j]) > threshold) {
                suppressed[j] = 1;
            }
        }
    }

    // Compact
    for (int i = 0; i < keep_count; i++) {
        dets[i] = dets[keep[i]];
    }
    *count = keep_count;
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;
    data = NULL;
    if (NULL == fp) return NULL;
    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0) return NULL;
    data = (unsigned char *)malloc(sz);
    if (data == NULL) return NULL;
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;
    fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    data = load_data(fp, 0, size);
    fclose(fp);
    *model_size = size;
    return data;
}

void* init_yolo26_model(const char* model_path) {
    int ret;
    RKNN_Context* ctx = (RKNN_Context*)malloc(sizeof(RKNN_Context));
    memset(ctx, 0, sizeof(RKNN_Context));

    int model_data_size = 0;
    unsigned char* model_data = load_model(model_path, &model_data_size);
    if (model_data == NULL) {
        free(ctx);
        return NULL;
    }

    ret = rknn_init(&ctx->ctx, model_data, model_data_size, 0, NULL);
    free(model_data);
    if (ret < 0) {
        printf("rknn_init failed! ret=%d\n", ret);
        free(ctx);
        return NULL;
    }

    ret = rknn_query(ctx->ctx, RKNN_QUERY_IN_OUT_NUM, &ctx->io_num, sizeof(ctx->io_num));
    if (ret < 0) {
        free(ctx);
        return NULL;
    }

    ctx->input_attrs = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * ctx->io_num.n_input);
    ctx->output_attrs = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * ctx->io_num.n_output);

    for (int i = 0; i < ctx->io_num.n_input; i++) {
        ctx->input_attrs[i].index = i;
        rknn_query(ctx->ctx, RKNN_QUERY_INPUT_ATTR, &(ctx->input_attrs[i]), sizeof(rknn_tensor_attr));
    }
    for (int i = 0; i < ctx->io_num.n_output; i++) {
        ctx->output_attrs[i].index = i;
        rknn_query(ctx->ctx, RKNN_QUERY_OUTPUT_ATTR, &(ctx->output_attrs[i]), sizeof(rknn_tensor_attr));
    }

    if (ctx->input_attrs[0].fmt == RKNN_TENSOR_NHWC) {
        ctx->model_width = ctx->input_attrs[0].dims[2];
        ctx->model_height = ctx->input_attrs[0].dims[1];
    } else {
        ctx->model_width = ctx->input_attrs[0].dims[3];
        ctx->model_height = ctx->input_attrs[0].dims[2];
    }

    ctx->is_init = true;
    return (void*)ctx;
}

void release_yolo26_model(void* ctx_ptr) {
    RKNN_Context* ctx = (RKNN_Context*)ctx_ptr;
    if (ctx && ctx->is_init) {
        rknn_destroy(ctx->ctx);
        if (ctx->input_attrs) free(ctx->input_attrs);
        if (ctx->output_attrs) free(ctx->output_attrs);
        free(ctx);
    }
}

int detect_yolo26(void* ctx_ptr, unsigned char* img_data, float conf_thres, float nms_thres, Detection26* results, int max_results) {
    RKNN_Context* ctx = (RKNN_Context*)ctx_ptr;
    if (!ctx || !ctx->is_init) return 0;

    int ret;
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = ctx->model_width * ctx->model_height * 3;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = img_data;

    ret = rknn_inputs_set(ctx->ctx, ctx->io_num.n_input, inputs);
    if (ret < 0) return 0;

    ret = rknn_run(ctx->ctx, NULL);
    if (ret < 0) return 0;

    rknn_output outputs[3];
    memset(outputs, 0, sizeof(outputs));
    for(int i=0; i<3; i++) outputs[i].want_float = 1;
    
    ret = rknn_outputs_get(ctx->ctx, ctx->io_num.n_output, outputs, NULL);
    if (ret < 0) return 0;

    int det_count = 0;

    // Process each scale (Stride 8, 16, 32)
    for (int i = 0; i < ctx->io_num.n_output && i < 3; i++) {
        float* data = (float*)outputs[i].buf;
        int h, w, c;
         
        // check attributes
        if (ctx->output_attrs[i].fmt == RKNN_TENSOR_NHWC) {
            h = ctx->output_attrs[i].dims[1];
            w = ctx->output_attrs[i].dims[2];
            c = ctx->output_attrs[i].dims[3];
        } else { // NCHW
            c = ctx->output_attrs[i].dims[1];
            h = ctx->output_attrs[i].dims[2];
            w = ctx->output_attrs[i].dims[3];
        }

        // Infer STRIDE
        int stride = ctx->model_width / w; // e.g. 640/80 = 8

        const float* current_anchors = NULL;
        if (stride == 8)  { current_anchors = &anchors[0][0]; }
        else if (stride == 16) { current_anchors = &anchors[1][0]; }
        else if (stride == 32) { current_anchors = &anchors[2][0]; }

        bool is_anchor_based = false;
        int num_anchors = 1;
        int dims_per_anchor = c; 
        int num_classes = 0;

        if (c == 6) {
             // 4 box + 2 class ?
             // Or 1 anchor * (4 + 1 + 1)?
             // Given user code 'nc: 2' -> 4 + 2 is most likely.
             is_anchor_based = false;
             num_classes = 2;
             dims_per_anchor = 6;
             num_anchors = 1; 
        } else if (c > 6) {
             if (c % 3 == 0) {
                 // Try 3 anchors assumption
                 int d = c / 3;
                 if (d >= 5) {
                     is_anchor_based = true;
                     num_anchors = 3;
                     dims_per_anchor = d;
                     num_classes = d - 5;
                 }
             }
             if (num_classes == 0) {
                 // Try 1 anchor / anchor-free
                 is_anchor_based = false; // assume anchor-free 4+NC
                 num_anchors = 1;
                 dims_per_anchor = c;
                 num_classes = c - 4; // no obj
             }
        } else {
             continue; // < 6 unsupported
        }

        // Iterate grid
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int a = 0; a < num_anchors; a++) {
                    
                    // Helper macro (only valid within this scope variables: ctx, offset, data, h, w, x, y, a, dims_per_anchor)
                    #define GET_VAL(ch_idx) ( \
                        (ctx->output_attrs[i].fmt == RKNN_TENSOR_NHWC) ? \
                        (data[offset + (ch_idx)]) : \
                        (data[(a * dims_per_anchor + (ch_idx)) * (h * w) + (y * w) + x]) \
                    )

                    int offset;
                    // For NHWC:
                    if (ctx->output_attrs[i].fmt == RKNN_TENSOR_NHWC) {
                         offset = y * w * c + x * c + a * dims_per_anchor;
                    } else {
                         offset = 0; // handled by macro
                    }

                    float final_score = 0;
                    int class_id = -1;

                    if (is_anchor_based) {
                        // Obj + Class
                        float obj_conf = sigmoid(GET_VAL(4));
                        if (obj_conf < conf_thres) continue;

                        float max_class_score = 0;
                        for (int cls = 0; cls < num_classes; cls++) {
                            float s = sigmoid(GET_VAL(5 + cls));
                            if (s > max_class_score) {
                                max_class_score = s;
                                class_id = cls;
                            }
                        }
                        final_score = obj_conf * max_class_score;
                    } else {
                        // Anchor-Free: No Obj, Just Class (4 + NC)
                        // Score = Max(Sigmoid(Class))
                        float max_class_score = 0;
                        for (int cls = 0; cls < num_classes; cls++) {
                            float s = sigmoid(GET_VAL(4 + cls));
                            if (s > max_class_score) {
                                max_class_score = s;
                                class_id = cls;
                            }
                        }
                        final_score = max_class_score;
                    }
                    
                    if (final_score < conf_thres) continue;

                    // Box decoding
                    float x1, y1, x2, y2;

                    if (is_anchor_based) {
                        // Standard YOLOv5/7
                        float vx = sigmoid(GET_VAL(0));
                        float vy = sigmoid(GET_VAL(1));
                        float vw = sigmoid(GET_VAL(2));
                        float vh = sigmoid(GET_VAL(3));

                        float bx = (vx * 2.0f - 0.5f + x) * stride;
                        float by = (vy * 2.0f - 0.5f + y) * stride;
                        float aw = anchors[i][2*a]; // Use anchor logic
                        float ah = anchors[i][2*a + 1];
                        float bw = powf(vw * 2.0f, 2.0f) * aw;
                        float bh = powf(vh * 2.0f, 2.0f) * ah;
                        
                        x1 = bx - bw * 0.5f;
                        y1 = by - bh * 0.5f;
                        x2 = bx + bw * 0.5f;
                        y2 = by + bh * 0.5f;
                    } else {
                        // Anchor-Free (YOLOv8/10) - LTRB Assumption
                        // values 0..3 are distances to l, t, r, b
                        // Need to know distribution. 
                        
                        float d0 = GET_VAL(0);
                        float d1 = GET_VAL(1);
                        float d2 = GET_VAL(2);
                        float d3 = GET_VAL(3);
                        
                        // Try DFL-style (but single value) -> Reg distances
                        // x1 = (x + 0.5 - d0) * stride
                        x1 = (x + 0.5f - d0) * stride;
                        y1 = (y + 0.5f - d1) * stride;
                        x2 = (x + 0.5f + d2) * stride;
                        y2 = (y + 0.5f + d3) * stride;
                    }

                    if (det_count < max_results) {
                        results[det_count].x1 = x1;
                        results[det_count].y1 = y1;
                        results[det_count].x2 = x2;
                        results[det_count].y2 = y2;
                        results[det_count].score = final_score;
                        results[det_count].class_id = class_id;
                        det_count++;
                    }
                    
                    #undef GET_VAL
                }
            }
        }
    }
    
    rknn_outputs_release(ctx->ctx, ctx->io_num.n_output, outputs);

    // NMS
    nms_process(results, &det_count, nms_thres);

    return det_count;
}
