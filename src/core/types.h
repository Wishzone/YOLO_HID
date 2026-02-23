#ifndef TYPES_H
#define TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float x1, y1, x2, y2;
    float score;
    int class_id;
} Detection;

#ifdef __cplusplus
}
#endif

#endif // TYPES_H
