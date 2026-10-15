#ifndef HID_MOTION_H
#define HID_MOTION_H

#include <algorithm>
#include <cmath>

// Shared by the real USB worker and the offline simulator. A successful report
// consumes this integer from the float buffer; a failed report consumes nothing.
inline int hid_axis_step(float value, float smooth_factor) {
    if (std::abs(value) <= 0.5f) return 0;
    const float smoothing = std::max(0.01f,std::min(1.0f,smooth_factor));
    float amount = value * smoothing;
    if (std::abs(amount) < 1.0f) amount = value > 0 ? 1.0f : -1.0f;
    return static_cast<int>(std::max(-127.0f,std::min(127.0f,amount)));
}

#endif
