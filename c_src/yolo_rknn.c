#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <sys/time.h>
#include <rknn_api.h>

#define MAX_DETECTIONS 100
// NUM_CLASS is now dynamic
// #define NUM_CLASS 80 
#define NUM_ANCHORS 8400

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

    printf("C: Loading model from %s\n", model_path);

    // Load model
    int model_data_size = 0;
    unsigned char* model_data = load_model(model_path, &model_data_size);
    if (model_data == NULL) {
        printf("Failed to load model file: %s\n", model_path);
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
    printf("C: rknn_init success\n");

    // Query IO
    ret = rknn_query(ctx->ctx, RKNN_QUERY_IN_OUT_NUM, &ctx->io_num, sizeof(ctx->io_num));
    if (ret < 0) {
        printf("rknn_query IO failed\n");
        free(ctx);
        return NULL;
    }

    ctx->input_attrs = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * ctx->io_num.n_input);
    ctx->output_attrs = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * ctx->io_num.n_output);

    for (int i = 0; i < ctx->io_num.n_input; i++) {
        ctx->input_attrs[i].index = i;
        ret = rknn_query(ctx->ctx, RKNN_QUERY_INPUT_ATTR, &(ctx->input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) printf("query input attr %d failed\n", i);
    }
    for (int i = 0; i < ctx->io_num.n_output; i++) {
        ctx->output_attrs[i].index = i;
        ret = rknn_query(ctx->ctx, RKNN_QUERY_OUTPUT_ATTR, &(ctx->output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) printf("query output attr %d failed\n", i);
    }


    // Assume input 0 is the image
    printf("Input 0 fmt: %d (NHWC=%d, NCHW=%d)\n", ctx->input_attrs[0].fmt, RKNN_TENSOR_NHWC, RKNN_TENSOR_NCHW);
    printf("Input 0 type: %d (UINT8=%d, INT8=%d, FLOAT=%d)\n", ctx->input_attrs[0].type, RKNN_TENSOR_UINT8, RKNN_TENSOR_INT8, RKNN_TENSOR_FLOAT32);
    printf("Input 0 ZP: %d, Scale: %f\n", ctx->input_attrs[0].zp, ctx->input_attrs[0].scale);

    if (ctx->input_attrs[0].fmt == RKNN_TENSOR_NHWC) {
        ctx->model_width = ctx->input_attrs[0].dims[2];
        ctx->model_height = ctx->input_attrs[0].dims[1];
    } else {
        ctx->model_width = ctx->input_attrs[0].dims[3];
        ctx->model_height = ctx->input_attrs[0].dims[2];
    }
    
    printf("C Model Init: %dx%d\n", ctx->model_width, ctx->model_height);
    
    // Print Output Dims and Attributes
    for (int i = 0; i < ctx->io_num.n_output; i++) {
        printf("Output %d dims: %d %d %d %d\n", i, 
            ctx->output_attrs[i].dims[0],
            ctx->output_attrs[i].dims[1],
            ctx->output_attrs[i].dims[2],
            ctx->output_attrs[i].dims[3]);
        printf("Output %d ZP: %d, Scale: %f\n", i, ctx->output_attrs[i].zp, ctx->output_attrs[i].scale);
    }

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

    static int frame_count = 0;
    frame_count++;
    bool debug_print = (frame_count % 60 == 0); // Print every 60 frames

    struct timeval start_time, stop_time;
    double time_diff;

    int ret;
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = ctx->model_width * ctx->model_height * 3;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = img_data;

    // Debug Input Data
    if (debug_print) {
        printf("Input Center Pixel: R=%d G=%d B=%d\n", 
            img_data[ctx->model_height/2 * ctx->model_width * 3 + ctx->model_width/2 * 3 + 0],
            img_data[ctx->model_height/2 * ctx->model_width * 3 + ctx->model_width/2 * 3 + 1],
            img_data[ctx->model_height/2 * ctx->model_width * 3 + ctx->model_width/2 * 3 + 2]);
    }

    ret = rknn_inputs_set(ctx->ctx, ctx->io_num.n_input, inputs);
    if (ret < 0) {
        printf("rknn_inputs_set failed\n");
        return 0;
    }

    gettimeofday(&start_time, NULL);
    ret = rknn_run(ctx->ctx, NULL);
    gettimeofday(&stop_time, NULL);
    if (debug_print) {
        printf("rknn_run: %f ms\n", (stop_time.tv_sec - start_time.tv_sec) * 1000.0 + (stop_time.tv_usec - start_time.tv_usec) / 1000.0);
    }

    if (ret < 0) {
        printf("rknn_run failed\n");
        return 0;
    }

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    
    ret = rknn_outputs_get(ctx->ctx, ctx->io_num.n_output, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get failed\n");
        return 0;
    }

    // Post Process
    float* output_data = (float*)outputs[0].buf;
    
    // Determine shape from attributes or output
    // We assume single output for YOLOv8/11
    int dim0 = ctx->output_attrs[0].dims[0]; // Batch (1)
    int dim1 = ctx->output_attrs[0].dims[1]; // 84 or 8400
    int dim2 = ctx->output_attrs[0].dims[2]; // 8400 or 84
    
    int rows, cols;
    bool transposed = false;

    // Standard YOLOv8 export: [1, 84, 8400] -> 84 rows (features), 8400 cols (anchors)
    // We want to iterate over anchors (cols)
    
    if (dim1 < dim2) {
        rows = dim1; // 84 (or 6)
        cols = dim2; // 8400
    } else {
        rows = dim1; // 8400
        cols = dim2; // 84 (or 6)
        transposed = true; // Need to handle [1, 8400, 84]
    }
    
    int num_classes;
    if (!transposed) {
        num_classes = rows - 4;
    } else {
        num_classes = cols - 4;
    }

    // Safety check
    if (num_classes <= 0) {
            printf("Unexpected output shape: rows=%d, cols=%d, transposed=%d\n", rows, cols, transposed);
            rknn_outputs_release(ctx->ctx, ctx->io_num.n_output, outputs);
            return 0;
    }

    int det_count = 0;
    Detection local_dets[MAX_DETECTIONS];

    gettimeofday(&start_time, NULL);

    if (!transposed) {
        // Shape: [rows, cols] e.g. [6, 8400]
        // Data layout: row0_col0, row0_col1, ... row1_col0 ...
        // We want to access: for each anchor i (0..8399)
        // cx = data[0 * cols + i]
        // cy = data[1 * cols + i]
        // ...
        // score = data[(4+c) * cols + i]
        
        for (int i = 0; i < cols; i++) {
            // Find max class score
            float max_score = 0;
            int class_id = -1;
            
            for (int c = 0; c < num_classes; c++) {
                float score = output_data[(4 + c) * cols + i];
                if (score > max_score) {
                    max_score = score;
                    class_id = c;
                }
            }

            // Debug print for the first few anchors
            if (debug_print && i < 5) {
                printf("Anchor %d: cx=%f, cy=%f, w=%f, h=%f, s0=%f, s1=%f\n", 
                       i, 
                       output_data[0*cols+i], 
                       output_data[1*cols+i], 
                       output_data[2*cols+i], 
                       output_data[3*cols+i], 
                       output_data[4*cols+i], 
                       output_data[5*cols+i]);
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
        // Shape: [rows, cols] e.g. [8400, 6]
        // Data layout: anchor0_feat0, anchor0_feat1 ...
        // stride = cols (6)
        int stride = cols; 
        int num_anchors = rows; // 8400
        
        for (int i = 0; i < num_anchors; i++) {
            float* ptr = output_data + i * stride;
            
            float max_score = 0;
            int class_id = -1;
            
            for (int c = 0; c < num_classes; c++) {
                float score = ptr[4 + c];
                if (score > max_score) {
                    max_score = score;
                    class_id = c;
                }
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

    rknn_outputs_release(ctx->ctx, ctx->io_num.n_output, outputs);

    // NMS
    nms(local_dets, &det_count, nms_thres);
    
    gettimeofday(&stop_time, NULL);
    if (debug_print) {
        printf("PostProcess: %f ms, Dets: %d\n", (stop_time.tv_sec - start_time.tv_sec) * 1000.0 + (stop_time.tv_usec - start_time.tv_usec) / 1000.0, det_count);
    }

    // Copy to output buffer
    int final_count = (det_count < max_results) ? det_count : max_results;
    for(int i=0; i<final_count; i++) {
        results[i] = local_dets[i];
    }
    
    return final_count;
}
