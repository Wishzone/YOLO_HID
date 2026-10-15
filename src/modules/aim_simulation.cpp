#include "aim_simulation.h"
#include "hid_motion.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>

bool validate_simulation(const SimulationOptions& o, std::string& error) {
    auto range = [&](double value, double low, double high, const char* name) {
        if (std::isfinite(value) && value >= low && value <= high) return true;
        error = std::string(name) + " must be in [" + std::to_string(low) + ", " + std::to_string(high) + "]";
        return false;
    };
    if (!range(o.width,64,7680,"width") || !range(o.height,64,4320,"height") ||
        !range(o.frames,1,2000,"frames") || !range(o.frame_ms,1,100,"frame-ms") ||
        !range(o.feedback_ms,0,500,"feedback-ms") || !range(o.hid_interval_ms,0.25,20,"hid-interval-ms") ||
        !range(o.offset_x,-o.width,o.width,"offset-x") || !range(o.offset_y,-o.height,o.height,"offset-y") ||
        !range(o.box_width,1,o.width,"box-width") || !range(o.box_height,1,o.height,"box-height") ||
        !range(o.velocity_x,-5000,5000,"vx") || !range(o.velocity_y,-5000,5000,"vy") ||
        !range(o.gain_x,0.001f,100,"gain-x") || !range(o.gain_y,0.001f,100,"gain-y") ||
        !range(o.smoothing,0.01f,1,"hid-smoothing") || !range(o.prediction_ms,0,40,"prediction-ms") ||
        !range(o.aim_ratio,0,1,"aim-ratio") || !range(o.settle_frames,1,2001,"settle-frames") ||
        !range(o.tolerance,0,100,"tolerance") || !range(o.control.kp,0,5,"kp") ||
        !range(o.control.kd,0,5,"kd") || !range(o.control.sensitivity,0.001f,10,"sensitivity") ||
        !range(o.control.deadzone,0,100,"deadzone")) return false;
    return true;
}

SimulationResult simulate_aim(const SimulationOptions& o) {
    std::string error;
    if (!validate_simulation(o,error)) throw std::invalid_argument(error);
    SimulationResult result; result.options = o;
    TrackerOptions tracking;
    tracking.body_aim_ratio = o.aim_ratio;
    tracking.prediction_ms = o.prediction_ms;
    if (o.head) tracking.head_classes = {0};
    TargetTracker tracker(tracking);
    AimController controller(o.control);
    struct Observation { Detection box; double captured, ready; };
    std::deque<Observation> observations;
    float camera_x = 0, camera_y = 0, buffer_x = 0, buffer_y = 0;
    AimTarget selected;
    AimControl control;
    bool fire = false, last_fire = false, initial_release = true;
    int captured = 0, recorded = 0, hid_tick = 0;
    int interval_dx = 0, interval_dy = 0, interval_reports = 0;
    const double infinity = std::numeric_limits<double>::infinity();
    auto truth = [&](double time) {
        const float x = o.width*0.5f + o.offset_x + o.velocity_x*time*0.001 - camera_x;
        const float y = o.height*0.5f + o.offset_y + o.velocity_y*time*0.001 - camera_y;
        return Detection{x-o.box_width*0.5f,y-o.box_height*0.5f,
                         x+o.box_width*0.5f,y+o.box_height*0.5f,1.0f,0};
    };
    auto record = [&](int frame) {
        SimulationFrame row;
        row.frame = frame; row.time_ms = frame*o.frame_ms; row.box = truth(row.time_ms);
        row.target = selected; row.control = control;
        row.camera_x = camera_x; row.camera_y = camera_y;
        row.error_x = (row.box.x1+row.box.x2)*0.5f-o.width*0.5f;
        row.error_y = row.box.y1+(row.box.y2-row.box.y1)*(o.head ? 0.5f : o.aim_ratio)-o.height*0.5f;
        row.buffer_x = buffer_x; row.buffer_y = buffer_y;
        row.hid_dx = interval_dx; row.hid_dy = interval_dy; row.reports = interval_reports;
        row.inside = row.box.x1 <= o.width*0.5f && row.box.x2 >= o.width*0.5f &&
                     row.box.y1 <= o.height*0.5f && row.box.y2 >= o.height*0.5f;
        row.fire = last_fire;
        result.trace.push_back(row);
        interval_dx = interval_dy = interval_reports = 0;
    };
    // Frame 0 is the instant the target appears, before the first control output.
    record(0);
    while (recorded < o.frames) {
        const double capture_time = captured < o.frames ? captured*o.frame_ms : infinity;
        const double ready_time = observations.empty() ? infinity : observations.front().ready;
        const double report_time = hid_tick*o.hid_interval_ms;
        const double record_time = (recorded+1)*o.frame_ms;
        const double now = std::min(std::min(capture_time,ready_time),std::min(report_time,record_time));
        const double epsilon = 1e-7;
        // Boundary records finish the preceding interval, before a new detection
        // or report at the same timestamp. Thus N frames always means N*frame_ms.
        if (record_time <= now+epsilon) {
            record(++recorded);
            if (recorded == o.frames) break;
        }
        if (capture_time <= now+epsilon) {
            observations.push_back({truth(now),now,now+o.feedback_ms});
            ++captured;
        }
        while (!observations.empty() && observations.front().ready <= now+epsilon) {
            const Observation sample = observations.front(); observations.pop_front();
            selected = tracker.update(&sample.box,1,1,0,0,o.width,o.height,sample.captured*0.001,AimMode::Nearest);
            if (selected.changed || !selected.visible) {
                buffer_x = buffer_y = 0; fire = false;
                controller.reset();
            }
            control = controller.update(selected,o.width,o.height);
            buffer_x += control.move_x; buffer_y += control.move_y; fire = control.fire;
        }
        if (report_time <= now+epsilon) {
            const int dx = initial_release ? 0 : hid_axis_step(buffer_x,o.smoothing);
            const int dy = initial_release ? 0 : hid_axis_step(buffer_y,o.smoothing);
            const bool next_fire = !initial_release && fire;
            if (initial_release || dx || dy || next_fire != last_fire) {
                buffer_x -= dx; buffer_y -= dy;
                camera_x += dx*o.gain_x; camera_y += dy*o.gain_y;
                interval_dx += dx; interval_dy += dy; ++interval_reports; ++result.reports;
                last_fire = next_fire; initial_release = false;
            }
            ++hid_tick;
        }
    }
    int inside_frames = 0, run = 0;
    bool previous_inside = false;
    for (const auto& row : result.trace) {
        if (row.inside) {
            if (result.first_entry_frame < 0) result.first_entry_frame = row.frame;
            if (++run >= o.settle_frames && result.settled_frame < 0)
                result.settled_frame = row.frame-o.settle_frames+1;
        } else {
            run = 0;
            if (previous_inside) ++result.exit_count;
        }
        previous_inside = row.inside;
        if (result.first_aim_frame < 0 && std::hypot(row.error_x,row.error_y) <= o.tolerance)
            result.first_aim_frame = row.frame;
        if (row.frame > 0) {
            if (row.inside) ++inside_frames;
            if (row.fire && !row.inside) ++result.outside_fire_frames;
        }
    }
    result.inside_ratio = double(inside_frames)/o.frames;
    const size_t tail_count = std::min(size_t(25),result.trace.size()-1);
    for (size_t i=result.trace.size()-tail_count; i<result.trace.size(); ++i)
        result.tail_error += std::hypot(result.trace[i].error_x,result.trace[i].error_y)/tail_count;
    return result;
}
