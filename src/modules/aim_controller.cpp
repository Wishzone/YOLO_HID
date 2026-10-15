#include "aim_controller.h"
#include <cmath>

void AimController::reset() {
    last_x_ = last_y_ = 0;
    tracking_ = false;
}

AimControl AimController::update(const AimTarget& target, int width, int height) {
    AimControl result;
    if (!target.visible || width <= 0 || height <= 0 ||
        !std::isfinite(target.x) || !std::isfinite(target.y)) {
        reset();
        return result;
    }
    if (target.changed) reset();
    float dx = target.x - width * 0.5f;
    float dy = target.y - height * 0.5f;
    if (std::abs(dx) < options_.deadzone) dx = 0;
    if (std::abs(dy) < options_.deadzone) dy = 0;
    result.error_x = dx; result.error_y = dy;
    result.move_x = (options_.kp * dx + (tracking_ ? options_.kd * (dx-last_x_) : 0)) * options_.sensitivity;
    result.move_y = (options_.kp * dy + (tracking_ ? options_.kd * (dy-last_y_) : 0)) * options_.sensitivity;
    last_x_ = dx; last_y_ = dy; tracking_ = true;
    const float distance = std::hypot(dx,dy);
    const float predicted_distance = std::hypot(dx*(1-options_.kp),dy*(1-options_.kp));
    result.fire = distance < options_.fire_radius ||
                  (predicted_distance < options_.fire_radius && distance < options_.fire_radius * 2.5f);
    return result;
}
