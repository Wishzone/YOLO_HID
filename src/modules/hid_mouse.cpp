#include "hid_mouse.h"
#include "../core/globals.h"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/poll.h>
#include <cmath>
#include <algorithm>
#include <thread>

static int hid_fd = -1;

void hid_init(const char* device_path) {
    hid_fd = open(device_path, O_RDWR | O_NONBLOCK);
    if (hid_fd < 0) std::cerr << "[HID] Error opening " << device_path << std::endl;
}

void hid_worker(float smooth_factor) {
    struct pollfd pfd;
    pfd.fd = hid_fd;
    pfd.events = POLLOUT;
    bool last_fire_state = false;
    
    while (running) {
        if (hid_fd < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(hid_mutex);
            if (std::abs(hid_buffer_x) < 0.5f && std::abs(hid_buffer_y) < 0.5f && fire_request == last_fire_state) {
                hid_cv.wait(lock);
            }
        }
        
        if (!running) break;
        if (poll(&pfd, 1, 5) <= 0) continue; 

        int dx = 0, dy = 0;
        bool current_fire = fire_request;

        {
            std::lock_guard<std::mutex> lock(hid_mutex);
            float step_x = hid_buffer_x * smooth_factor;
            float step_y = hid_buffer_y * smooth_factor;
            
            // Min movement logic
            if (std::abs(hid_buffer_x) > 0.5f && std::abs(step_x) < 1.0f) 
                step_x = (hid_buffer_x > 0) ? 1.0f : -1.0f;
            else if (std::abs(hid_buffer_x) <= 0.5f) step_x = 0;
                
            if (std::abs(hid_buffer_y) > 0.5f && std::abs(step_y) < 1.0f) 
                step_y = (hid_buffer_y > 0) ? 1.0f : -1.0f;
            else if (std::abs(hid_buffer_y) <= 0.5f) step_y = 0;
            
            step_x = std::max(-127.0f, std::min(127.0f, step_x));
            step_y = std::max(-127.0f, std::min(127.0f, step_y));
            
            dx = (int)step_x;
            dy = (int)step_y;
            
            if (dx != 0 || dy != 0 || current_fire != last_fire_state) {
                uint8_t report[] = {(uint8_t)(current_fire ? 0x01 : 0), (uint8_t)dx, (uint8_t)dy, 0}; 
                if (write(hid_fd, report, 4) == 4) {
                    hid_buffer_x -= dx;
                    hid_buffer_y -= dy;
                }
                last_fire_state = current_fire;
            }
        }
    }
}
