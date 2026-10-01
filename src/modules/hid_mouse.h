#ifndef HID_MOUSE_H
#define HID_MOUSE_H

#include <cstdint>
#include <string>

struct HidStatus {
    bool opened;
    uint64_t reports;
    uint64_t errors;
};

bool hid_init(const char* device_path);
void hid_worker(float smooth_factor);
HidStatus hid_get_status();
std::string hid_usb_state();

#endif // HID_MOUSE_H
