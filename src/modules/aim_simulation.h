#ifndef AIM_SIMULATION_H
#define AIM_SIMULATION_H

#include "aim_controller.h"
#include <string>
#include <vector>

struct SimulationOptions {
    ControlOptions control;
    int width = 1920, height = 1080, frames = 150;
    double frame_ms = 20, feedback_ms = 0, hid_interval_ms = 1;
    float offset_x = 300, offset_y = 0, box_width = 80, box_height = 120;
    float velocity_x = 0, velocity_y = 0; // Independent target motion, source pixels/s.
    float gain_x = 1, gain_y = 1; // Observed source pixels per HID count.
    float smoothing = 1, prediction_ms = 0, aim_ratio = 0.2f;
    bool head = false;
    int settle_frames = 5;
    float tolerance = 2;
};

struct SimulationFrame {
    int frame = 0;
    double time_ms = 0;
    Detection box{}; // Current truth on screen; deliberately not clipped.
    AimTarget target; // Last processed (possibly delayed) detection + prediction.
    AimControl control;
    float camera_x = 0, camera_y = 0, error_x = 0, error_y = 0;
    float buffer_x = 0, buffer_y = 0;
    int hid_dx = 0, hid_dy = 0, reports = 0;
    bool inside = false, fire = false;
};

struct SimulationResult {
    SimulationOptions options;
    std::vector<SimulationFrame> trace;
    int first_entry_frame = -1, settled_frame = -1, first_aim_frame = -1;
    int exit_count = 0, reports = 0, outside_fire_frames = 0;
    double inside_ratio = 0, tail_error = 0;
};

// Pure offline calculation. Never opens a camera, model, USB device or socket.
SimulationResult simulate_aim(const SimulationOptions& options);
bool validate_simulation(const SimulationOptions& options, std::string& error);

#endif
