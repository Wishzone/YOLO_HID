#include "../src/modules/aim_simulation.h"
#include "../src/modules/hid_motion.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>

static void near(float actual, float expected) {
    if (std::abs(actual-expected) >= 0.001f) std::fprintf(stderr,"Expected %.4f, got %.4f\n",expected,actual);
    assert(std::abs(actual-expected) < 0.001f);
}

int main() {
    // Independent checks of the pre-refactor, per-frame PD equation.
    AimController controller;
    AimTarget target; target.visible = true; target.x = 1260; target.y = 504;
    auto command = controller.update(target,1920,1080);
    near(command.move_x,90); near(command.move_y,-10.8f); assert(!command.fire);
    target.x = 1160; target.y = 520;
    command = controller.update(target,1920,1080);
    near(command.move_x,42.5f); near(command.move_y,-3.2f);
    target.changed = true;
    command = controller.update(target,1920,1080);
    near(command.move_x,60); near(command.move_y,-6);
    target.changed = false; target.visible = false;
    near(controller.update(target,1920,1080).move_x,0);
    target.visible = true;
    near(controller.update(target,1920,1080).move_x,60);
    target.x = 970; target.y = 540;
    assert(controller.update(target,1920,1080).fire);

    assert(hid_axis_step(0.5f,1) == 0 && hid_axis_step(-0.5f,1) == 0);
    assert(hid_axis_step(0.51f,1) == 1 && hid_axis_step(-0.51f,1) == -1);
    assert(hid_axis_step(300,1) == 127 && hid_axis_step(-300,1) == -127);
    assert(hid_axis_step(9.9f,0.5f) == 4 && hid_axis_step(-9.9f,0.5f) == -4);

    SimulationOptions options;
    auto normal = simulate_aim(options);
    assert(normal.trace.size() == 151 && normal.trace.front().frame == 0);
    near(normal.trace.front().camera_x,0);
    // 300px initial error -> .6*.5*300=90 HID units. A camera pan
    // moves the observed target in the opposite direction.
    near(normal.trace[1].camera_x,90);
    near(normal.trace[1].error_x,210);
    near(normal.trace[1].box.x1,1130);
    assert(normal.first_entry_frame > 0 && normal.settled_frame >= normal.first_entry_frame);
    assert(normal.exit_count == 0 && normal.tail_error <= options.tolerance);
    assert(normal.trace.back().time_ms == 3000);
    assert(!normal.trace[normal.first_entry_frame-1].inside);
    for (int i=0; i<options.settle_frames; ++i) assert(normal.trace[normal.settled_frame+i].inside);

    // An explicit USB slot at 20ms cannot be counted in the interval [0,20).
    options.hid_interval_ms = 20;
    auto slow_usb = simulate_aim(options);
    near(slow_usb.trace[1].camera_x,0);
    // The second stale frame adds another 90 counts before the first moving
    // report; the real worker then emits its per-axis maximum of 127.
    near(slow_usb.trace[2].camera_x,127);

    options = SimulationOptions(); options.feedback_ms = 20;
    auto delayed = simulate_aim(options);
    near(delayed.trace[1].camera_x,0);
    near(delayed.trace[2].camera_x,90);
    options = SimulationOptions(); options.smoothing = 0.01f;
    auto smoothed = simulate_aim(options);
    near(smoothed.trace[1].camera_x,19);
    assert(smoothed.trace[1].buffer_x > 60);

    // Fractional commands accumulate, rather than vanishing through int casts.
    options = SimulationOptions(); options.control.kp = 0.01f; options.control.kd = 0;
    options.control.sensitivity = 0.001f; options.frames = 200;
    assert(simulate_aim(options).trace.back().camera_x > 0);
    options = SimulationOptions(); options.control.kp = options.control.kd = 0;
    auto still = simulate_aim(options);
    assert(still.first_entry_frame == -1 && still.settled_frame == -1 && still.first_aim_frame == -1);
    near(still.trace.back().camera_x,0);
    options.offset_x = 0;
    auto already_inside = simulate_aim(options);
    assert(already_inside.first_entry_frame == 0 && already_inside.settled_frame == 0);

    // Source-pixel gain is separate from the controller's output multiplier.
    options = SimulationOptions(); options.gain_x = 2;
    near(simulate_aim(options).trace[1].camera_x,180);
    options = SimulationOptions(); options.frames = 1;
    assert(simulate_aim(options).settled_frame == -1);
    options = SimulationOptions(); options.head = true;
    near(simulate_aim(options).trace[0].error_y,0);

    // Exercise actual tracker prediction under delayed moving-target feedback.
    options = SimulationOptions(); options.velocity_x = 160; options.feedback_ms = 20;
    auto plain = simulate_aim(options);
    options.prediction_ms = 20;
    auto predicted = simulate_aim(options);
    bool different = false;
    for (size_t i=0; i<plain.trace.size(); ++i) {
        if (std::abs(plain.trace[i].target.x-predicted.trace[i].target.x) > 0.01f) different = true;
        assert(std::isfinite(predicted.trace[i].error_x));
    }
    assert(different);
    std::string error;
    options.frame_ms = 0; assert(!validate_simulation(options,error));
    options.frame_ms = std::numeric_limits<double>::quiet_NaN(); assert(!validate_simulation(options,error));
    puts("PASS Simulation: original PD, HID quantization, closed-loop feedback, delay, USB cadence, prediction and entry metrics");
}
