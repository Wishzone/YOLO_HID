#ifndef HID_MOUSE_H
#define HID_MOUSE_H

void hid_init(const char* device_path);
void hid_worker(float smooth_factor);

#endif // HID_MOUSE_H
