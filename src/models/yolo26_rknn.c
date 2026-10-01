#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>
#include <rknn_api.h>
#include "yolo26_rknn.h"
#include "yolo26_postprocess.h"

// Supports raw YOLO26 feature maps and end-to-end XYXY detections.

typedef struct {
    rknn_context ctx;
    bool is_init;
    bool raw_outputs;
    int scores_mode;
    float nms_iou;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_width;
    int model_height;
} RKNN_Context;

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
    if (fread(data, 1, sz, fp) != sz) {
        free(data);
        return NULL;
    }
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
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long length = ftell(fp);
    if (length <= 0 || length > INT_MAX) {
        fclose(fp);
        return NULL;
    }
    int size = (int)length;
    data = load_data(fp, 0, size);
    fclose(fp);
    *model_size = size;
    return data;
}

void* init_yolo26_model(const char* model_path) {
    int ret;
    RKNN_Context* ctx = (RKNN_Context*)calloc(1, sizeof(RKNN_Context));
    if (ctx == NULL) return NULL;
    ctx->nms_iou = 0.5f;

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
    ctx->is_init = true;

    ret = rknn_query(ctx->ctx, RKNN_QUERY_IN_OUT_NUM, &ctx->io_num, sizeof(ctx->io_num));
    if (ret < 0 || ctx->io_num.n_input != 1 || ctx->io_num.n_output < 1 || ctx->io_num.n_output > 3) {
        fprintf(stderr, "YOLO26 requires one input and one to three detection outputs.\n");
        release_yolo26_model(ctx);
        return NULL;
    }

    ctx->input_attrs = (rknn_tensor_attr*)calloc(ctx->io_num.n_input, sizeof(rknn_tensor_attr));
    ctx->output_attrs = (rknn_tensor_attr*)calloc(ctx->io_num.n_output, sizeof(rknn_tensor_attr));
    if (!ctx->input_attrs || !ctx->output_attrs) {
        release_yolo26_model(ctx);
        return NULL;
    }

    for (int i = 0; i < ctx->io_num.n_input; i++) {
        ctx->input_attrs[i].index = i;
        if (rknn_query(ctx->ctx, RKNN_QUERY_INPUT_ATTR, &(ctx->input_attrs[i]), sizeof(rknn_tensor_attr)) < 0) {
            release_yolo26_model(ctx);
            return NULL;
        }
    }
    for (int i = 0; i < ctx->io_num.n_output; i++) {
        ctx->output_attrs[i].index = i;
        if (rknn_query(ctx->ctx, RKNN_QUERY_OUTPUT_ATTR, &(ctx->output_attrs[i]), sizeof(rknn_tensor_attr)) < 0) {
            release_yolo26_model(ctx);
            return NULL;
        }
    }

    if (ctx->input_attrs[0].n_dims != 4) {
        release_yolo26_model(ctx);
        return NULL;
    }
    int channels;
    if (ctx->input_attrs[0].fmt == RKNN_TENSOR_NHWC) {
        ctx->model_width = ctx->input_attrs[0].dims[2];
        ctx->model_height = ctx->input_attrs[0].dims[1];
        channels = ctx->input_attrs[0].dims[3];
    } else {
        ctx->model_width = ctx->input_attrs[0].dims[3];
        ctx->model_height = ctx->input_attrs[0].dims[2];
        channels = ctx->input_attrs[0].dims[1];
    }
    if (ctx->model_width != 640 || ctx->model_height != 640 || channels != 3) {
        fprintf(stderr, "YOLO26 preprocessing requires a 640x640 RGB input.\n");
        release_yolo26_model(ctx);
        return NULL;
    }
    const rknn_tensor_attr* output = &ctx->output_attrs[0];
    ctx->raw_outputs = output->n_dims == 4;
    bool valid = true;
    if (ctx->raw_outputs) {
        int expected_channels = 0;
        for (unsigned i = 0; i < ctx->io_num.n_output; ++i) {
            const rknn_tensor_attr* attr = &ctx->output_attrs[i];
            bool nhwc = attr->fmt == RKNN_TENSOR_NHWC;
            unsigned c = attr->dims[nhwc ? 3 : 1];
            unsigned h = attr->dims[nhwc ? 1 : 2];
            unsigned w = attr->dims[nhwc ? 2 : 3];
            if (i == 0) expected_channels = c;
            if (attr->n_dims != 4 || attr->dims[0] != 1 ||
                (attr->fmt != RKNN_TENSOR_NHWC && attr->fmt != RKNN_TENSOR_NCHW) ||
                c <= 4 || c > 1024 || h == 0 || h > 640 || w == 0 || w > 640 ||
                c != (unsigned)expected_channels || attr->n_elems != c * h * w) valid = false;
        }
    } else {
        valid = ctx->io_num.n_output == 1 &&
                ((output->n_dims == 2 && (output->dims[0] == 6 || output->dims[1] == 6)) ||
                 (output->n_dims == 3 && output->dims[0] == 1 &&
                  (output->dims[1] == 6 || output->dims[2] == 6)));
    }
    if (!valid) {
        fprintf(stderr, "Unsupported YOLO26 output layout.\n");
        release_yolo26_model(ctx);
        return NULL;
    }
    printf("[Model] %s, outputs=%u, layout=%s\n", model_path,
           ctx->io_num.n_output, ctx->raw_outputs ? "raw feature maps" : "end-to-end detections");

    return (void*)ctx;
}

void release_yolo26_model(void* ctx_ptr) {
    RKNN_Context* ctx = (RKNN_Context*)ctx_ptr;
    if (ctx) {
        if (ctx->is_init) rknn_destroy(ctx->ctx);
        if (ctx->input_attrs) free(ctx->input_attrs);
        if (ctx->output_attrs) free(ctx->output_attrs);
        free(ctx);
    }
}

void set_yolo26_scores_mode(void* ctx_ptr, int mode) {
    RKNN_Context* ctx = (RKNN_Context*)ctx_ptr;
    if (ctx && mode >= 0 && mode <= 2) ctx->scores_mode = mode;
}

void set_yolo26_nms_iou(void* ctx_ptr, float iou) {
    RKNN_Context* ctx = (RKNN_Context*)ctx_ptr;
    if (ctx && isfinite(iou) && iou > 0.0f && iou < 1.0f) ctx->nms_iou = iou;
}

int detect_yolo26(void* ctx_ptr, unsigned char* img_data, float conf_thres, Detection* results, int max_results) {
    RKNN_Context* ctx = (RKNN_Context*)ctx_ptr;
    if (!ctx || !ctx->is_init || !img_data || !results || max_results <= 0) return -1;
    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.type = RKNN_TENSOR_UINT8;
    input.size = ctx->model_width * ctx->model_height * 3;
    input.fmt = RKNN_TENSOR_NHWC;
    input.buf = img_data;
    if (rknn_inputs_set(ctx->ctx, 1, &input) < 0 || rknn_run(ctx->ctx, NULL) < 0) return -1;

    rknn_output outputs[3];
    memset(outputs, 0, sizeof(outputs));
    for (unsigned i = 0; i < ctx->io_num.n_output; ++i) {
        outputs[i].index = i;
        outputs[i].want_float = 1;
    }
    if (rknn_outputs_get(ctx->ctx, ctx->io_num.n_output, outputs, NULL) < 0) return -1;
    for (unsigned i = 0; i < ctx->io_num.n_output; ++i) {
        if (!outputs[i].buf || outputs[i].size < ctx->output_attrs[i].n_elems * sizeof(float)) {
            rknn_outputs_release(ctx->ctx, ctx->io_num.n_output, outputs);
            return -1;
        }
    }
    int count;
    if (ctx->raw_outputs) {
        Yolo26RawHead heads[3];
        for (unsigned i = 0; i < ctx->io_num.n_output; ++i) {
            const rknn_tensor_attr* attr = &ctx->output_attrs[i];
            bool nhwc = attr->fmt == RKNN_TENSOR_NHWC;
            heads[i].data = (const float*)outputs[i].buf;
            heads[i].channels = attr->dims[nhwc ? 3 : 1];
            heads[i].height = attr->dims[nhwc ? 1 : 2];
            heads[i].width = attr->dims[nhwc ? 2 : 3];
            heads[i].is_nhwc = nhwc;
            heads[i].scores_mode = ctx->scores_mode;
        }
        count = yolo26_decode_raw_heads_nms(heads, ctx->io_num.n_output,
                                       ctx->model_width, ctx->model_height,
                                       conf_thres, ctx->nms_iou, results, max_results);
    } else {
        const rknn_tensor_attr* attr = &ctx->output_attrs[0];
        unsigned last = attr->n_dims - 1;
        bool box_last = attr->dims[last] == 6;
        int rows = box_last ? attr->dims[last - 1] : attr->dims[last];
        count = yolo26_decode_detections((const float*)outputs[0].buf, rows, box_last,
                                        ctx->model_width, ctx->model_height,
                                        conf_thres, results, max_results);
    }
    rknn_outputs_release(ctx->ctx, ctx->io_num.n_output, outputs);
    return count;
}
