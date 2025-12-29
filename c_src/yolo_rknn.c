#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <sys/time.h>
#include <rknn_api.h>

#define MAX_DETECTIONS 100
#define DFL_REG 16

typedef struct {
    float x1, y1, x2, y2;
    float score;
    int class_id;
} Detection;

typedef struct {
    rknn_context ctx;
    bool is_init;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_width;
    int model_height;
} RKNN_Context;

// Softmax for DFL
float dfl_softmax_integral(float* src) {
    float exp_sum = 0.0f;
    float exps[DFL_REG];
    float max_val = -10000.0f;
    
    for(int i=0; i<DFL_REG; i++) {
        if(src[i] > max_val) max_val = src[i];
    }
    
    for(int i=0; i<DFL_REG; i++) {
        exps[i] = expf(src[i] - max_val);
        exp_sum += exps[i];
    }
    
    float result = 0.0f;
    for(int i=0; i<DFL_REG; i++) {
        result += (exps[i] / exp_sum) * i;
    }
    return result;
}

// Sigmoid
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// NMS Helper
float iou(Detection* a, Detection* b) {
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

void nms(Detection* dets, int* count, float threshold) {
    int keep[MAX_DETECTIONS];
    int keep_count = 0;
    int suppressed[MAX_DETECTIONS] = {0};

    // Sort by score descending
    for (int i = 0; i < *count - 1; i++) {
        for (int j = 0; j < *count - i - 1; j++) {
            if (dets[j].score < dets[j + 1].score) {
                Detection temp = dets[j];
                dets[j] = dets[j + 1];
                dets[j + 1] = temp;
            }
        }
    }

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

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
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

void* init_model(const char* model_path) {
    int ret;
    // Ensure stdout is unbuffered for debugging
    setbuf(stdout, NULL);

    RKNN_Context* ctx = (RKNN_Context*)malloc(sizeof(RKNN_Context));
    memset(ctx, 0, sizeof(RKNN_Context));


    // Load model
    int model_data_size = 0;
    unsigned char* model_data = load_model(model_path, &model_data_size);
    if (model_data == NULL) {
        printf("[RKNN] Model load failed: %s\n", model_path);
        free(ctx);
        return NULL;
    }

    ret = rknn_init(&ctx->ctx, model_data, model_data_size, 0, NULL);
    free(model_data);
    if (ret < 0) {
        printf("[RKNN] rknn_init failed! ret=%d\n", ret);
        free(ctx);
        return NULL;
    }

    // Query IO
    ret = rknn_query(ctx->ctx, RKNN_QUERY_IN_OUT_NUM, &ctx->io_num, sizeof(ctx->io_num));
    if (ret < 0) {
        printf("[RKNN] rknn_query IO failed\n");
        free(ctx);
        return NULL;
    }

    ctx->input_attrs = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * ctx->io_num.n_input);
    ctx->output_attrs = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * ctx->io_num.n_output);

    for (int i = 0; i < ctx->io_num.n_input; i++) {
        ctx->input_attrs[i].index = i;
        ret = rknn_query(ctx->ctx, RKNN_QUERY_INPUT_ATTR, &(ctx->input_attrs[i]), sizeof(rknn_tensor_attr));
    }
    for (int i = 0; i < ctx->io_num.n_output; i++) {
        ctx->output_attrs[i].index = i;
        ret = rknn_query(ctx->ctx, RKNN_QUERY_OUTPUT_ATTR, &(ctx->output_attrs[i]), sizeof(rknn_tensor_attr));
    }

    if (ctx->input_attrs[0].fmt == RKNN_TENSOR_NHWC) {
        ctx->model_width = ctx->input_attrs[0].dims[2];
        ctx->model_height = ctx->input_attrs[0].dims[1];
    } else {
        ctx->model_width = ctx->input_attrs[0].dims[3];
        ctx->model_height = ctx->input_attrs[0].dims[2];
    }

    printf("[RKNN] Model Ready: %dx%d\n", ctx->model_width, ctx->model_height);

    ctx->is_init = true;
    return (void*)ctx;
}

void release_model(void* ctx_ptr) {
    RKNN_Context* ctx = (RKNN_Context*)ctx_ptr;
    if (ctx && ctx->is_init) {
        rknn_destroy(ctx->ctx);
        if (ctx->input_attrs) free(ctx->input_attrs);
        if (ctx->output_attrs) free(ctx->output_attrs);
        free(ctx);
    }
}

// Returns number of detections
int detect(void* ctx_ptr, unsigned char* img_data, float conf_thres, float nms_thres, Detection* results, int max_results) {
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
    if (ret < 0) {
        printf("rknn_inputs_set failed\n");
        return 0;
    }

    ret = rknn_run(ctx->ctx, NULL);
    if (ret < 0) {
        printf("rknn_run failed\n");
        return 0;
    }

    // Support up to 3 outputs
    rknn_output outputs[3];
    memset(outputs, 0, sizeof(outputs));
    for(int i=0; i<ctx->io_num.n_output; i++) {
        outputs[i].want_float = 1;
    }
    
    ret = rknn_outputs_get(ctx->ctx, ctx->io_num.n_output, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get failed\n");
        return 0;
    }

    int det_count = 0;
    Detection local_dets[MAX_DETECTIONS];

    // Check if it's a multi-head output (3 outputs)
    if (ctx->io_num.n_output == 3) {
        for (int i = 0; i < 3; i++) {
            float* output_data = (float*)outputs[i].buf;
            int h, w, c;
            int fmt = ctx->output_attrs[i].fmt;
            
            // Robust dimension detection
            if (fmt == RKNN_TENSOR_NHWC) {
                h = ctx->output_attrs[i].dims[1];
                w = ctx->output_attrs[i].dims[2];
                c = ctx->output_attrs[i].dims[3];
            } else {
                // NCHW
                c = ctx->output_attrs[i].dims[1];
                h = ctx->output_attrs[i].dims[2];
                w = ctx->output_attrs[i].dims[3];
            }
            
            // Calculate stride dynamically based on model width (640)
            // Assuming square model
            int stride = ctx->model_width / w;

            // Debug first frame only
            static bool printed_info = false;
            if (!printed_info) {
                printf("[RKNN] Output %d: fmt=%d (0=NCHW, 1=NHWC), dims=[%d, %d, %d, %d] -> h=%d, w=%d, c=%d, stride=%d\n", 
                       i, fmt, 
                       ctx->output_attrs[i].dims[0], ctx->output_attrs[i].dims[1], 
                       ctx->output_attrs[i].dims[2], ctx->output_attrs[i].dims[3],
                       h, w, c, stride);
            }

            // Check if dimensions match expectation
            if (c != 66) {
                // Try to adapt if it's not 66
                // If c is 80 or 40, maybe we parsed dims wrong?
                if (h == 66) { // Swap H and C?
                     int temp = h; h = c; c = temp;
                     // If we swapped, we might need to change reading logic, but let's assume fmt was wrong
                     // But let's just warn for now
                     if (!printed_info) printf("[RKNN] Warning: c=%d, expected 66. Maybe dims are swapped?\n", c);
                }
            }
            int num_classes = c - 64;
            
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    // Find max class score first to filter
                    float max_score = 0;
                    int class_id = -1;
                    
                    // Class scores start at offset 64
                    for (int cls = 0; cls < num_classes; cls++) {
                        float score;
                        if (fmt == RKNN_TENSOR_NHWC) {
                            score = output_data[y * w * c + x * c + (64 + cls)];
                        } else {
                            score = output_data[(64 + cls) * h * w + y * w + x];
                        }
                        
                        score = sigmoid(score); // Apply sigmoid
                        if (score > max_score) {
                            max_score = score;
                            class_id = cls;
                        }
                    }
                    
                    if (max_score > conf_thres) {
                        if (det_count >= MAX_DETECTIONS) break;
                        
                        // Decode DFL
                        float dist[4]; // left, top, right, bottom
                        float dfl_buf[16];
                        
                        for(int d=0; d<4; d++) {
                            for(int k=0; k<16; k++) {
                                if (fmt == RKNN_TENSOR_NHWC) {
                                    dfl_buf[k] = output_data[y * w * c + x * c + (d * 16 + k)];
                                } else {
                                    dfl_buf[k] = output_data[(d * 16 + k) * h * w + y * w + x];
                                }
                            }
                            dist[d] = dfl_softmax_integral(dfl_buf);
                        }
                        
                        float anchor_cx = (x + 0.5f) * stride;
                        float anchor_cy = (y + 0.5f) * stride;
                        
                        float x1 = anchor_cx - dist[0] * stride;
                        float y1 = anchor_cy - dist[1] * stride;
                        float x2 = anchor_cx + dist[2] * stride;
                        float y2 = anchor_cy + dist[3] * stride;
                        
                        local_dets[det_count].x1 = x1;
                        local_dets[det_count].y1 = y1;
                        local_dets[det_count].x2 = x2;
                        local_dets[det_count].y2 = y2;
                        local_dets[det_count].score = max_score;
                        local_dets[det_count].class_id = class_id;
                        det_count++;

                        if (!printed_info && det_count < 5) {
                             printf("[RKNN] Det: cls=%d, score=%.2f, box=[%.1f, %.1f, %.1f, %.1f] grid=(%d,%d) stride=%d\n",
                                    class_id, max_score, x1, y1, x2, y2, x, y, stride);
                        }
                    }
                }
            }
            if (!printed_info && i == 2) printed_info = true;
        }
    } else {
        // Fallback to single output logic (if user switches back)
        // ... (Existing logic for single output)
        float* output_data = (float*)outputs[0].buf;
        int dim1 = ctx->output_attrs[0].dims[1];
        int dim2 = ctx->output_attrs[0].dims[2];
        int rows, cols;
        bool transposed = false;
        if (dim1 < dim2) { rows = dim1; cols = dim2; } 
        else { rows = dim1; cols = dim2; transposed = true; }
        int num_classes = (!transposed ? rows : cols) - 4;
        
        if (num_classes > 0) {
             if (!transposed) {
                for (int i = 0; i < cols; i++) {
                    float max_score = 0;
                    int class_id = -1;
                    for (int c = 0; c < num_classes; c++) {
                        float score = output_data[(4 + c) * cols + i];
                        if (score > max_score) { max_score = score; class_id = c; }
                    }
                    if (max_score > conf_thres) {
                        if (det_count >= MAX_DETECTIONS) break;
                        float cx = output_data[0 * cols + i];
                        float cy = output_data[1 * cols + i];
                        float w = output_data[2 * cols + i];
                        float h = output_data[3 * cols + i];
                        local_dets[det_count].x1 = cx - w / 2;
                        local_dets[det_count].y1 = cy - h / 2;
                        local_dets[det_count].x2 = cx + w / 2;
                        local_dets[det_count].y2 = cy + h / 2;
                        local_dets[det_count].score = max_score;
                        local_dets[det_count].class_id = class_id;
                        det_count++;
                    }
                }
            } else {
                int stride = cols; 
                int num_anchors = rows;
                for (int i = 0; i < num_anchors; i++) {
                    float* ptr = output_data + i * stride;
                    float max_score = 0;
                    int class_id = -1;
                    for (int c = 0; c < num_classes; c++) {
                        float score = ptr[4 + c];
                        if (score > max_score) { max_score = score; class_id = c; }
                    }
                    if (max_score > conf_thres) {
                        if (det_count >= MAX_DETECTIONS) break;
                        float cx = ptr[0];
                        float cy = ptr[1];
                        float w = ptr[2];
                        float h = ptr[3];
                        local_dets[det_count].x1 = cx - w / 2;
                        local_dets[det_count].y1 = cy - h / 2;
                        local_dets[det_count].x2 = cx + w / 2;
                        local_dets[det_count].y2 = cy + h / 2;
                        local_dets[det_count].score = max_score;
                        local_dets[det_count].class_id = class_id;
                        det_count++;
                    }
                }
            }
        }
    }

    rknn_outputs_release(ctx->ctx, ctx->io_num.n_output, outputs);

    // NMS
    nms(local_dets, &det_count, nms_thres);
    
    // Copy to output buffer
    int final_count = (det_count < max_results) ? det_count : max_results;
    for(int i=0; i<final_count; i++) {
        results[i] = local_dets[i];
    }
    
    return final_count;
}

