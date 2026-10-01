#include "yolo26_postprocess.h"
#include <limits.h>
#include <math.h>

static float clamp_coordinate(float value, int maximum) {
    return fmaxf(0.0f, fminf(value, (float)maximum));
}

static float probability(float logit) {
    if (logit >= 0.0f) return 1.0f / (1.0f + expf(-logit));
    float value = expf(logit);
    return value / (1.0f + value);
}

static float head_value(const Yolo26RawHead* head, int position, int channel) {
    int index = head->is_nhwc ? position * head->channels + channel
                             : channel * head->height * head->width + position;
    return head->data[index];
}

int yolo26_decode_raw_heads(const Yolo26RawHead* heads, int head_count,
                           int input_width, int input_height, float threshold,
                           Detection* results, int capacity) {
    if (!heads || head_count <= 0 || !results || capacity <= 0 ||
        input_width <= 0 || input_height <= 0) return -1;
    for (int index = 0; index < head_count; ++index) {
        if (!heads[index].data || heads[index].channels <= 4 ||
            heads[index].height <= 0 || heads[index].width <= 0) return -1;
    }

    int count = 0;
    int scores_are_logits = heads[0].scores_mode == 1;
    if (heads[0].scores_mode == 0) {
        for (int index = 0; index < head_count && !scores_are_logits; ++index) {
            const Yolo26RawHead* head = &heads[index];
            for (int position = 0; position < head->height * head->width && !scores_are_logits; ++position) {
                for (int channel = 4; channel < head->channels; ++channel) {
                    float value = head_value(head, position, channel);
                    if (isfinite(value) && (value < 0.0f || value > 1.0f)) {
                        scores_are_logits = 1;
                        break;
                    }
                }
            }
        }
    }
    for (int index = 0; index < head_count; ++index) {
        const Yolo26RawHead* head = &heads[index];
        float stride_x = (float)input_width / head->width;
        float stride_y = (float)input_height / head->height;
        for (int y = 0; y < head->height; ++y) {
            for (int x = 0; x < head->width; ++x) {
                int position = y * head->width + x;
                float score = 0.0f;
                int class_id = -1;
                for (int channel = 4; channel < head->channels; ++channel) {
                    float logit = head_value(head, position, channel);
                    if (!isfinite(logit)) continue;
                    float candidate = scores_are_logits ? probability(logit) : logit;
                    if (candidate < 0.0f || candidate > 1.0f) continue;
                    if (candidate > score) {
                        score = candidate;
                        class_id = channel - 4;
                    }
                }
                if (class_id < 0 || score < threshold) continue;
                float left = head_value(head, position, 0);
                float top = head_value(head, position, 1);
                float right = head_value(head, position, 2);
                float bottom = head_value(head, position, 3);
                if (!isfinite(left) || !isfinite(top) || !isfinite(right) || !isfinite(bottom)) continue;
                Detection detection;
                detection.x1 = clamp_coordinate((x + 0.5f - left) * stride_x, input_width);
                detection.y1 = clamp_coordinate((y + 0.5f - top) * stride_y, input_height);
                detection.x2 = clamp_coordinate((x + 0.5f + right) * stride_x, input_width);
                detection.y2 = clamp_coordinate((y + 0.5f + bottom) * stride_y, input_height);
                detection.score = score;
                detection.class_id = class_id;
                if (detection.x2 <= detection.x1 || detection.y2 <= detection.y1) continue;
                if (count < capacity) results[count++] = detection;
            }
        }
    }
    // Keep the original no-NMS policy.
    return count;
}

int yolo26_decode_detections(const float* data, int rows, int box_last,
                            int input_width, int input_height, float threshold,
                            Detection* results, int capacity) {
    if (!data || rows <= 0 || !results || capacity <= 0 ||
        input_width <= 0 || input_height <= 0) return -1;
    float maximum = 0.0f;
    for (int row = 0; row < rows; ++row) {
        float x2 = data[box_last ? row * 6 + 2 : 2 * rows + row];
        float y2 = data[box_last ? row * 6 + 3 : 3 * rows + row];
        if (isfinite(x2) && x2 > maximum) maximum = x2;
        if (isfinite(y2) && y2 > maximum) maximum = y2;
    }
    float sx = maximum <= 1.5f ? (float)input_width : 1.0f;
    float sy = maximum <= 1.5f ? (float)input_height : 1.0f;
    int count = 0;
    for (int row = 0; row < rows; ++row) {
        float values[6];
        int valid = 1;
        for (int column = 0; column < 6; ++column) {
            values[column] = data[box_last ? row * 6 + column : column * rows + row];
            if (!isfinite(values[column])) valid = 0;
        }
        if (!valid || values[4] < threshold || values[4] > 1.0f ||
            values[5] < 0.0f || values[5] >= (float)INT_MAX) continue;
        Detection detection;
        detection.x1 = clamp_coordinate(values[0] * sx, input_width);
        detection.y1 = clamp_coordinate(values[1] * sy, input_height);
        detection.x2 = clamp_coordinate(values[2] * sx, input_width);
        detection.y2 = clamp_coordinate(values[3] * sy, input_height);
        detection.score = values[4];
        detection.class_id = (int)values[5];
        if (detection.x2 <= detection.x1 || detection.y2 <= detection.y1) continue;
        if (count < capacity) results[count++] = detection;
    }
    return count;
}
