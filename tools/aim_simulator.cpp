#include "../src/modules/aim_simulation.h"
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

static void options_json(const SimulationOptions& o) {
    std::cout << "{\"width\":" << o.width << ",\"height\":" << o.height << ",\"frames\":" << o.frames
        << ",\"frame-ms\":" << o.frame_ms << ",\"feedback-ms\":" << o.feedback_ms << ",\"hid-interval-ms\":" << o.hid_interval_ms
        << ",\"offset-x\":" << o.offset_x << ",\"offset-y\":" << o.offset_y
        << ",\"box-width\":" << o.box_width << ",\"box-height\":" << o.box_height
        << ",\"vx\":" << o.velocity_x << ",\"vy\":" << o.velocity_y
        << ",\"gain-x\":" << o.gain_x << ",\"gain-y\":" << o.gain_y
        << ",\"kp\":" << o.control.kp << ",\"kd\":" << o.control.kd
        << ",\"sensitivity\":" << o.control.sensitivity << ",\"deadzone\":" << o.control.deadzone
        << ",\"hid-smoothing\":" << o.smoothing << ",\"prediction-ms\":" << o.prediction_ms
        << ",\"aim-ratio\":" << o.aim_ratio << ",\"head\":" << int(o.head)
        << ",\"settle-frames\":" << o.settle_frames << ",\"tolerance\":" << o.tolerance << '}';
}

static std::string runtime_command(const SimulationOptions& o) {
    std::ostringstream out;
    out << "./yolo_app_26 --kp " << o.control.kp << " --kd " << o.control.kd
        << " --sensitivity " << o.control.sensitivity << " --deadzone " << o.control.deadzone
        << " --hid-smoothing " << o.smoothing << " --prediction-ms " << o.prediction_ms
        << " --aim-ratio " << o.aim_ratio;
    return out.str();
}

static void print_json(const SimulationResult& r) {
    std::cout << "{\"options\":"; options_json(r.options);
    std::cout << ",\"summary\":{\"first_entry_frame\":" << r.first_entry_frame
        << ",\"settled_frame\":" << r.settled_frame << ",\"first_aim_frame\":" << r.first_aim_frame
        << ",\"exit_count\":" << r.exit_count << ",\"reports\":" << r.reports
        << ",\"inside_ratio\":" << r.inside_ratio << ",\"tail_error\":" << r.tail_error
        << ",\"outside_fire_frames\":" << r.outside_fire_frames
        << "},\"runtime_command\":\"" << runtime_command(r.options) << "\",\"trace\":[";
    bool first = true;
    for (const auto& row : r.trace) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << "{\"frame\":" << row.frame << ",\"time_ms\":" << row.time_ms
            << ",\"box\":[" << row.box.x1 << ',' << row.box.y1 << ',' << row.box.x2 << ',' << row.box.y2 << ']'
            << ",\"camera_x\":" << row.camera_x << ",\"camera_y\":" << row.camera_y
            << ",\"error_x\":" << row.error_x << ",\"error_y\":" << row.error_y
            << ",\"target_x\":" << row.target.x << ",\"target_y\":" << row.target.y
            << ",\"target_id\":" << row.target.id << ",\"visible\":" << (row.target.visible ? "true" : "false")
            << ",\"move_x\":" << row.control.move_x << ",\"move_y\":" << row.control.move_y
            << ",\"control_error_x\":" << row.control.error_x << ",\"control_error_y\":" << row.control.error_y
            << ",\"buffer_x\":" << row.buffer_x << ",\"buffer_y\":" << row.buffer_y
            << ",\"hid_dx\":" << row.hid_dx << ",\"hid_dy\":" << row.hid_dy << ",\"reports\":" << row.reports
            << ",\"inside\":" << (row.inside ? "true" : "false") << ",\"fire\":" << (row.fire ? "true" : "false") << '}';
    }
    std::cout << "]}\n";
}

static void print_csv(const SimulationResult& r) {
    std::cout << "frame,time_ms,box_x1,box_y1,box_x2,box_y2,camera_x,camera_y,error_x,error_y,"
        "target_x,target_y,target_id,visible,control_error_x,control_error_y,move_x,move_y,"
        "hid_dx,hid_dy,reports,buffer_x,buffer_y,inside,fire\n";
    for (const auto& row : r.trace) {
        std::cout << row.frame << ',' << row.time_ms << ',' << row.box.x1 << ',' << row.box.y1 << ',' << row.box.x2 << ',' << row.box.y2
            << ',' << row.camera_x << ',' << row.camera_y << ',' << row.error_x << ',' << row.error_y
            << ',' << row.target.x << ',' << row.target.y << ',' << row.target.id << ',' << row.target.visible
            << ',' << row.control.error_x << ',' << row.control.error_y << ',' << row.control.move_x << ',' << row.control.move_y
            << ',' << row.hid_dx << ',' << row.hid_dy << ',' << row.reports << ',' << row.buffer_x << ',' << row.buffer_y
            << ',' << row.inside << ',' << row.fire << '\n';
    }
}

int main(int argc, char** argv) {
    SimulationOptions options;
    std::string format = "text";
    struct Setter { std::function<void(double)> set; bool integer; };
    std::map<std::string,Setter> setters;
#define SIM_PARAM(name, field, integral) setters["--" name] = {[&](double n) { field = static_cast<decltype(field)>(n); }, integral}
    SIM_PARAM("width",options.width,true); SIM_PARAM("height",options.height,true);
    SIM_PARAM("frames",options.frames,true); SIM_PARAM("frame-ms",options.frame_ms,false);
    SIM_PARAM("feedback-ms",options.feedback_ms,false); SIM_PARAM("hid-interval-ms",options.hid_interval_ms,false);
    SIM_PARAM("offset-x",options.offset_x,false); SIM_PARAM("offset-y",options.offset_y,false);
    SIM_PARAM("box-width",options.box_width,false); SIM_PARAM("box-height",options.box_height,false);
    SIM_PARAM("vx",options.velocity_x,false); SIM_PARAM("vy",options.velocity_y,false);
    SIM_PARAM("gain-x",options.gain_x,false); SIM_PARAM("gain-y",options.gain_y,false);
    SIM_PARAM("kp",options.control.kp,false); SIM_PARAM("kd",options.control.kd,false);
    SIM_PARAM("sensitivity",options.control.sensitivity,false); SIM_PARAM("deadzone",options.control.deadzone,false);
    SIM_PARAM("hid-smoothing",options.smoothing,false); SIM_PARAM("prediction-ms",options.prediction_ms,false);
    SIM_PARAM("aim-ratio",options.aim_ratio,false); SIM_PARAM("head",options.head,true);
    SIM_PARAM("settle-frames",options.settle_frames,true); SIM_PARAM("tolerance",options.tolerance,false);
#undef SIM_PARAM
    try {
        for (int i=1; i<argc; ++i) {
            const std::string name = argv[i];
            if (name == "--help" || name == "-h") {
                std::cout << "Offline fixed-crosshair simulation (no USB/camera/model).\n"
                    "Usage: " << argv[0] << " [--format text|json|csv] [OPTION NUMBER ...]\n"
                    "Default: 1920x1080, target offset (300,0), box 80x120, 20ms/frame, kp=.6 kd=.35 sensitivity=.5.\n"
                    "--feedback-ms adds capture-to-control latency; --hid-interval-ms models USB report slots (default 1ms).\n"
                    "--gain-x/--gain-y: measured source pixels per HID count. --vx/--vy: target pixels/second.\n"
                    "--head 1 uses box center; otherwise --aim-ratio defaults to .2.\nOptions:\n";
                for (const auto& entry : setters) std::cout << "  " << entry.first << '\n';
                return 0;
            }
            if (i+1 >= argc) throw std::invalid_argument("Missing value for " + name);
            const std::string value = argv[++i];
            if (name == "--format") { format = value; continue; }
            const auto found = setters.find(name);
            if (found == setters.end()) throw std::invalid_argument("Unknown option: " + name);
            size_t consumed = 0;
            const double number = std::stod(value,&consumed);
            if (consumed != value.size() || !std::isfinite(number) || std::abs(number) > 1e8 ||
                (found->second.integer && number != std::floor(number)) ||
                (name == "--head" && number != 0 && number != 1)) throw std::invalid_argument("Invalid " + name);
            found->second.set(number);
        }
        if (format != "json" && format != "csv" && format != "text") throw std::invalid_argument("Unknown format: " + format);
        const auto result = simulate_aim(options);
        std::cout << std::setprecision(9);
        if (format == "json") print_json(result);
        else if (format == "csv") print_csv(result);
        else {
            auto metric = [&](const char* label, int frame) {
                std::cout << label << ": ";
                if (frame < 0) std::cout << "not reached\n";
                else std::cout << frame << " frames / " << frame*options.frame_ms << " ms\n";
            };
            metric("First entry",result.first_entry_frame);
            metric("Stable entry (consecutive in-box frames)",result.settled_frame);
            metric("Aim tolerance",result.first_aim_frame);
            std::cout << "Exits: " << result.exit_count << ", in-box: " << result.inside_ratio*100
                << "%, last 25-frame mean aim error: " << result.tail_error << " px, USB reports: " << result.reports << '\n'
                << "Runtime parameters: " << runtime_command(options) << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Simulation error: " << error.what() << '\n'; return 2;
    }
    return 0;
}
