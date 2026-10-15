#ifndef AIM_CONTROLLER_H
#define AIM_CONTROLLER_H

#include "target_tracker.h"

// Gains retain the original per-detection-frame PD definition (no dt division).
struct ControlOptions {
    float kp = 0.60f;
    float kd = 0.35f;
    float sensitivity = 0.5f;
    float deadzone = 0.0f;
    float fire_radius = 15.0f;
};

struct AimControl {
    float error_x = 0, error_y = 0;
    float move_x = 0, move_y = 0; // HID counts, before report quantization.
    bool fire = false;
};

class AimController {
public:
    explicit AimController(const ControlOptions& options = ControlOptions()) : options_(options) {}
    AimControl update(const AimTarget& target, int width, int height);
    void reset();
private:
    ControlOptions options_;
    float last_x_ = 0, last_y_ = 0;
    bool tracking_ = false;
};

#endif
