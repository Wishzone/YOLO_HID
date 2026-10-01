#include "../src/models/yolo26_postprocess.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

static void close_to(float actual, float expected) {
    assert(fabsf(actual - expected) < 0.001f);
}

int main(void) {
    Detection results[4];
    float planar[24];
    for (int i = 0; i < 24; ++i) planar[i] = -20.0f;
    planar[1] = 0.25f;
    planar[5] = 0.25f;
    planar[9] = 0.5f;
    planar[13] = 0.5f;
    planar[21] = 2.0f;
    Yolo26RawHead head = {planar, 6, 2, 2, 0, 1};
    assert(yolo26_decode_raw_heads(&head, 1, 640, 640, 0.45f, results, 4) == 1);
    close_to(results[0].x1, 400.0f);
    close_to(results[0].y1, 80.0f);
    close_to(results[0].x2, 640.0f);
    close_to(results[0].y2, 320.0f);
    close_to(results[0].score, 0.880797f);
    assert(results[0].class_id == 1);
    assert(yolo26_decode_raw_heads(&head, 1, 640, 640, 0.9f, results, 4) == 0);

    float interleaved[24];
    for (int position = 0; position < 4; ++position)
        for (int channel = 0; channel < 6; ++channel)
            interleaved[position * 6 + channel] = planar[channel * 4 + position];
    head.data = interleaved;
    head.is_nhwc = 1;
    assert(yolo26_decode_raw_heads(&head, 1, 640, 640, 0.45f, results, 4) == 1);
    close_to(results[0].x1, 400.0f);
    Yolo26RawHead heads[2] = {head, head};
    assert(yolo26_decode_raw_heads(heads, 2, 640, 640, 0.45f, results, 1) == 1);
    // Probability heads must not receive a second sigmoid (zero must stay zero).
    float probabilities[6] = {0.25f,0.25f,0.5f,0.5f,0.0f,0.0f};
    Yolo26RawHead probability_head = {probabilities, 6, 1, 1, 1, 0};
    assert(yolo26_decode_raw_heads(&probability_head, 1, 640, 640, 0.45f, results, 4) == 0);
    probabilities[5] = 0.8f;
    assert(yolo26_decode_raw_heads(&probability_head, 1, 640, 640, 0.45f, results, 4) == 1);
    close_to(results[0].score, 0.8f);

    float decoded[12] = {10,20,110,120,0.9f,1, 5,10,15,20,0.2f,0};
    assert(yolo26_decode_detections(decoded, 2, 1, 640, 640, 0.45f, results, 4) == 1);
    close_to(results[0].x1, 10.0f);
    close_to(results[0].y2, 120.0f);
    float transposed[12];
    for (int row = 0; row < 2; ++row)
        for (int column = 0; column < 6; ++column)
            transposed[column * 2 + row] = decoded[row * 6 + column];
    assert(yolo26_decode_detections(transposed, 2, 0, 640, 640, 0.45f, results, 4) == 1);
    close_to(results[0].x2, 110.0f);
    float normalized[6] = {0.1f,0.2f,0.4f,0.5f,0.8f,0};
    assert(yolo26_decode_detections(normalized, 1, 1, 640, 640, 0.45f, results, 4) == 1);
    close_to(results[0].x1, 64.0f);
    close_to(results[0].y2, 320.0f);
    assert(yolo26_decode_detections(NULL, 1, 1, 640, 640, 0.45f, results, 4) == -1);
    puts("POSTPROCESS_TESTS_PASS");
    return 0;
}
