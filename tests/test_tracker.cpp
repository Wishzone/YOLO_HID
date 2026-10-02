#include "../src/modules/target_tracker.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>

static Detection body(float x, float y=500) { return {x-20,y-20,x+20,y+80,0.9f,2}; }
static Detection head(float x, float y=500) { return {x-10,y-10,x+10,y+10,0.8f,3}; }
static AimTarget update(TargetTracker& tracker, const std::vector<Detection>& boxes, double time,
                        AimMode mode=AimMode::Nearest) {
    return tracker.update(boxes.data(),boxes.size(),1,0,0,1000,1000,time,mode);
}

int main() {
    TrackerOptions options; options.target_classes = {2,3}; options.head_classes = {3};
    TargetTracker tracker(options);
    auto first = update(tracker,{body(470),body(540)},0);
    assert(first.visible && first.changed && first.x == 470);
    // Detection ordering and a small distance advantage must not make selection oscillate.
    auto stable = update(tracker,{body(532),body(465)},0.02);
    assert(stable.id == first.id && !stable.changed && stable.x == 465);
    auto hold = update(tracker,{body(440),body(505)},0.05);
    assert(hold.id == first.id);
    auto switched = update(tracker,{body(440),body(505)},0.2);
    assert(switched.id != first.id && switched.changed && switched.x == 505);
    auto missed = update(tracker,{},0.22);
    assert(missed.id == switched.id && !missed.visible && missed.x == 0);
    auto reacquired = update(tracker,{body(507)},0.24);
    assert(reacquired.id == switched.id && reacquired.visible);
    auto expired = update(tracker,{},0.5);
    assert(expired.id == 0 && !expired.visible);

    // Two detections in one frame cannot both overwrite the same existing track.
    tracker.reset();
    first = update(tracker,{body(500)},1);
    stable = update(tracker,{body(505),body(500)},1.02);
    assert(stable.id == first.id && stable.x == 500);
    hold = update(tracker,{body(505)},1.04);
    assert(!hold.visible && hold.id == first.id);
    switched = update(tracker,{body(505)},1.2);
    assert(switched.visible && switched.id != first.id);

    tracker.reset();
    first = update(tracker,{body(500),head(520)},2);
    assert(first.class_id == 2);
    auto prioritized = update(tracker,{body(500),head(520)},2.02,AimMode::HeadPriority);
    assert(prioritized.class_id == 3 && prioritized.y == 500);
    assert(update(tracker,{body(500),head(520)},2.04,AimMode::BodyOnly).class_id == 2);
    assert(update(tracker,{body(500)},2.06,AimMode::HeadOnly).id == 0);
    assert(update(tracker,{head(520)},2.08,AimMode::HeadOnly).class_id == 3);

    // Own labels are explicit: an odd class is not inherently a head.
    TrackerOptions custom; custom.target_classes = {7,8}; custom.head_classes = {8};
    TargetTracker custom_tracker(custom);
    Detection a=body(500), b=head(520); a.class_id=7; b.class_id=8;
    assert(update(custom_tracker,{a,b},3,AimMode::HeadOnly).class_id == 8);
    assert(update(custom_tracker,{a,b},3.02,AimMode::BodyOnly).class_id == 7);

    TrackerOptions limited=options; limited.aim_radius=50;
    TargetTracker radius_tracker(limited);
    assert(!update(radius_tracker,{body(580)},4).visible);
    assert(update(radius_tracker,{body(530)},4.02).visible);

    // Undo real letterbox dimensions, not the reference script's fixed 416-pixel scale.
    tracker.reset();
    const Detection encoded{300,300,340,340,0.8f,3};
    auto mapped=tracker.update(&encoded,1,1.0f/3,0,140,1920,1080,5,AimMode::Nearest);
    assert(mapped.visible && std::abs(mapped.x-960)<0.01 && std::abs(mapped.y-540)<0.01);
    Detection invalid=body(500); invalid.x1=std::numeric_limits<float>::quiet_NaN();
    tracker.reset(); assert(!update(tracker,{invalid},6).visible);
    Detection padding{10,10,20,20,0.9f,3};
    assert(!tracker.update(&padding,1,1.0f/3,0,140,1920,1080,6.02,AimMode::Nearest).visible);

    // Prediction is bounded, disabled by default, and reset on a missing frame/reversal.
    TrackerOptions predicted=options; predicted.prediction_ms=40;
    TargetTracker prediction_tracker(predicted), no_prediction(options);
    AimTarget lead, plain;
    for (int i=0; i<5; ++i) {
        lead=update(prediction_tracker,{body(400+i*10)},7+i*0.02);
        plain=update(no_prediction,{body(400+i*10)},7+i*0.02);
    }
    assert(lead.x > plain.x && lead.x-plain.x <= 20.001f && plain.x == 440);
    auto reversed=update(prediction_tracker,{body(430)},7.1);
    assert(reversed.x <= 430 && reversed.x >= 410);
    assert(!update(prediction_tracker,{},7.12).visible);
    auto after_gap=update(prediction_tracker,{body(425)},7.14);
    assert(after_gap.visible && after_gap.x == 425);
    auto after_pause=update(prediction_tracker,{body(425)},8);
    assert(after_pause.visible && after_pause.changed && after_pause.id != after_gap.id);
    auto backwards=update(prediction_tracker,{body(425)},7.9);
    assert(backwards.changed && backwards.id != after_pause.id);

    AimMode mode;
    assert(parse_aim_mode("head-priority",mode) && mode == AimMode::HeadPriority);
    assert(!parse_aim_mode("head_priority",mode));
    puts("PASS Tracker: association, stable selection, dropout, own labels, modes, letterbox and bounded prediction");
}
