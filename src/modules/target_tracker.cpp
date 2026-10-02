#include "target_tracker.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

const char* aim_mode_name(AimMode mode) {
    switch (mode) {
    case AimMode::HeadPriority: return "head-priority";
    case AimMode::HeadOnly: return "head-only";
    case AimMode::BodyOnly: return "body-only";
    default: return "nearest";
    }
}
bool parse_aim_mode(const char* name, AimMode& mode) {
    for (AimMode candidate : {AimMode::Nearest, AimMode::HeadPriority, AimMode::HeadOnly, AimMode::BodyOnly})
        if (std::strcmp(name, aim_mode_name(candidate)) == 0) { mode = candidate; return true; }
    return false;
}
static bool contains(const std::vector<int>& ids, int id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}
static float center_x(const Detection& box) { return (box.x1 + box.x2) * 0.5f; }
static float center_y(const Detection& box) { return (box.y1 + box.y2) * 0.5f; }
static float diagonal(const Detection& box) { return std::hypot(box.x2-box.x1, box.y2-box.y1); }
static float overlap(const Detection& a, const Detection& b) {
    const float area = std::max(0.0f, std::min(a.x2,b.x2)-std::max(a.x1,b.x1)) *
                       std::max(0.0f, std::min(a.y2,b.y2)-std::max(a.y1,b.y1));
    const float total = (a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-area;
    return total > 0 ? area/total : 0;
}

TargetTracker::TargetTracker(const TrackerOptions& options) : options_(options) {}
void TargetTracker::reset() {
    tracks_.clear(); locked_id_ = 0; last_switch_ = last_time_ = -1;
    last_width_ = last_height_ = 0;
}

AimTarget TargetTracker::update(const Detection* detections, int count, float r, int dw, int dh,
                                int width, int height, double now, AimMode mode) {
    AimTarget result;
    if (!std::isfinite(now) || !std::isfinite(r) || r <= 0 || width <= 0 || height <= 0 ||
        count < 0 || (count > 0 && !detections)) { reset(); return result; }
    const uint64_t previous_id = locked_id_;
    const double previous_time = last_time_;
    const bool discontinuity = last_time_ >= 0 && (now < last_time_ || now-last_time_ > 0.5);
    if (discontinuity || (last_width_ && (width != last_width_ || height != last_height_))) reset();
    if (mode != last_mode_) { locked_id_ = 0; last_switch_ = -1; }
    last_time_ = now; last_mode_ = mode; last_width_ = width; last_height_ = height;
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const Track& t) {
        return now-t.seen > std::max(0.25, options_.missing_grace);
    }), tracks_.end());
    for (auto& track : tracks_) track.visible = false;

    std::vector<Detection> boxes;
    for (int i=0; i<count; ++i) {
        Detection box = detections[i];
        if (box.class_id < 0 || (!options_.target_classes.empty() && !contains(options_.target_classes, box.class_id)) ||
            !std::isfinite(box.x1) || !std::isfinite(box.y1) || !std::isfinite(box.x2) ||
            !std::isfinite(box.y2) || !std::isfinite(box.score) || box.score < 0 || box.score > 1 ||
            box.x2 <= box.x1 || box.y2 <= box.y1) continue;
        box.x1 = std::max(0.0f, std::min(float(width), (box.x1-dw)/r));
        box.y1 = std::max(0.0f, std::min(float(height), (box.y1-dh)/r));
        box.x2 = std::max(0.0f, std::min(float(width), (box.x2-dw)/r));
        box.y2 = std::max(0.0f, std::min(float(height), (box.y2-dh)/r));
        if (box.x2 > box.x1 && box.y2 > box.y1) boxes.push_back(box);
    }
    struct Edge { size_t track, box; float cost; };
    std::vector<Edge> edges;
    for (size_t i=0; i<tracks_.size(); ++i) for (size_t j=0; j<boxes.size(); ++j) {
        const auto& track = tracks_[i]; const auto& box = boxes[j];
        if (track.box.class_id != box.class_id) continue;
        const float size_ratio = (box.x2-box.x1)*(box.y2-box.y1) /
                                ((track.box.x2-track.box.x1)*(track.box.y2-track.box.y1));
        if (size_ratio < 0.25f || size_ratio > 4) continue;
        const float distance = std::hypot(center_x(box)-center_x(track.box), center_y(box)-center_y(track.box));
        const float gate = std::max(20.0f, std::min(std::hypot(float(width),float(height))*0.08f,
                                                 std::max(diagonal(box),diagonal(track.box))*0.75f));
        if (distance > gate) continue;
        edges.push_back({i,j, distance/gate + 1-overlap(box,track.box)});
    }
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        if (a.cost != b.cost) return a.cost < b.cost;
        if (a.track != b.track) return a.track < b.track;
        return a.box < b.box;
    });
    std::vector<bool> used_track(tracks_.size(),false), used_box(boxes.size(),false);
    for (const auto& edge : edges) {
        if (used_track[edge.track] || used_box[edge.box]) continue;
        used_track[edge.track] = true; used_box[edge.box] = true;
        auto& track = tracks_[edge.track]; const auto& box = boxes[edge.box];
        const double dt = now-track.seen;
        if (dt > 0.0001 && dt < 0.1 && track.seen >= previous_time-0.000001) {
            const float vx = (center_x(box)-center_x(track.box))/dt;
            const float vy = (center_y(box)-center_y(track.box))/dt;
            const float alpha = 1-std::exp(-float(dt)/0.06f);
            if (vx*track.vx+vy*track.vy < 0) { track.vx = vx*0.25f; track.vy = vy*0.25f; }
            else { track.vx += alpha*(vx-track.vx); track.vy += alpha*(vy-track.vy); }
            ++track.hits;
        } else { track.vx = track.vy = 0; track.hits = 1; }
        track.box = box; track.seen = now; track.visible = true;
    }
    for (size_t i=0; i<boxes.size(); ++i) if (!used_box[i]) tracks_.emplace_back(next_id_++,boxes[i],now);

    const float normalizer = std::hypot(float(width),float(height))*0.5f;
    auto aim_y = [&](const Track& t) {
        return t.box.y1+(t.box.y2-t.box.y1)*(contains(options_.head_classes,t.box.class_id) ? 0.5f : options_.body_aim_ratio);
    };
    auto eligible = [&](const Track& t) {
        const bool head = contains(options_.head_classes,t.box.class_id);
        if ((mode == AimMode::HeadOnly && !head) || (mode == AimMode::BodyOnly && head)) return false;
        return options_.aim_radius <= 0 || std::hypot(center_x(t.box)-width*0.5f,aim_y(t)-height*0.5f) <= options_.aim_radius;
    };
    auto cost = [&](const Track& t) {
        const float distance = std::hypot(center_x(t.box)-width*0.5f,aim_y(t)-height*0.5f)/normalizer;
        return distance-(mode == AimMode::HeadPriority && contains(options_.head_classes,t.box.class_id) ? 0.12f : 0);
    };
    Track* best = nullptr; Track* locked = nullptr;
    for (auto& track : tracks_) {
        if (!eligible(track)) continue;
        if (track.id == locked_id_) locked = &track;
        if (track.visible && (!best || cost(track) < cost(*best) ||
            (cost(track) == cost(*best) && track.box.score > best->box.score))) best = &track;
    }
    if (locked && !locked->visible && now-locked->seen <= options_.missing_grace) {
        result.id = locked->id; result.class_id = locked->box.class_id;
        // Identity survives a short dropout, but there is no current point to control.
        result.changed = result.id != previous_id;
        return result;
    }
    Track* chosen = best;
    if (locked && locked->visible && best && best->id != locked->id &&
        (now-last_switch_ < options_.switch_cooldown || cost(*best)+0.025f >= cost(*locked))) chosen = locked;
    if (!chosen) { locked_id_ = 0; result.changed = previous_id != 0; return result; }
    if (chosen->id != locked_id_) last_switch_ = now;
    locked_id_ = chosen->id;
    result.id = chosen->id; result.class_id = chosen->box.class_id; result.visible = true;
    result.changed = result.id != previous_id || discontinuity;
    result.box = chosen->box; result.x = center_x(chosen->box); result.y = aim_y(*chosen);
    if (options_.prediction_ms > 0 && chosen->hits >= 4) {
        float px = chosen->vx*std::min(40.0f,options_.prediction_ms)*0.001f;
        float py = chosen->vy*std::min(40.0f,options_.prediction_ms)*0.001f;
        const float length = std::hypot(px,py);
        const float limit = std::min(20.0f,diagonal(chosen->box)*0.25f);
        if (length > limit) { px *= limit/length; py *= limit/length; }
        result.x += px; result.y += py;
    }
    result.x = std::max(0.0f,std::min(float(width-1),result.x));
    result.y = std::max(0.0f,std::min(float(height-1),result.y));
    return result;
}
