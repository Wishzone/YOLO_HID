#ifndef TARGET_TRACKER_H
#define TARGET_TRACKER_H

#include "../core/types.h"
#include <cstdint>
#include <vector>

enum class AimMode { Nearest, HeadPriority, HeadOnly, BodyOnly };
const char* aim_mode_name(AimMode mode);
bool parse_aim_mode(const char* name, AimMode& mode);

struct TrackerOptions {
    std::vector<int> target_classes;
    std::vector<int> head_classes;
    float body_aim_ratio = 0.2f;
    float aim_radius = 0; // Source pixels, zero allows the full frame.
    float prediction_ms = 0; // Optional lead; never extrapolate a missing detection.
    double missing_grace = 0.12;
    double switch_cooldown = 0.15;
};

struct AimTarget {
    uint64_t id = 0;
    int class_id = -1;
    bool visible = false;
    bool changed = false;
    float x = 0, y = 0; // Source-frame pixels, after optional bounded prediction.
    Detection box{};    // Source-frame box from the current detection.
};

// Lightweight geometric association, not a person/re-identification tracker.
class TargetTracker {
public:
    explicit TargetTracker(const TrackerOptions& options);
    AimTarget update(const Detection* detections, int count, float r, int dw, int dh,
                     int width, int height, double timestamp, AimMode mode);
    void reset();
private:
    struct Track {
        uint64_t id;
        Detection box;
        double seen;
        float vx = 0, vy = 0;
        unsigned hits = 1;
        bool visible = true;
        Track(uint64_t value, const Detection& detection, double time)
            : id(value), box(detection), seen(time) {}
    };
    TrackerOptions options_;
    std::vector<Track> tracks_;
    uint64_t next_id_ = 1, locked_id_ = 0;
    double last_switch_ = -1, last_time_ = -1;
    AimMode last_mode_ = AimMode::Nearest;
    int last_width_ = 0, last_height_ = 0;
};

#endif
