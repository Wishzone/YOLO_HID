#include "hid_mouse.h"
#include "../core/globals.h"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/poll.h>
#include <glob.h>
#include <fstream>
#include <cerrno>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>

static int hid_fd = -1;
static std::string hid_path;
static std::atomic<bool> hid_opened(false);
static std::atomic<uint64_t> hid_reports(0), hid_errors(0);

bool hid_init(const char* device_path) {
    hid_path = device_path;
    hid_fd = open(device_path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    hid_opened = hid_fd >= 0;
    if (hid_fd < 0)
        std::cerr << "[HID] Cannot open " << device_path << ": " << strerror(errno) << std::endl;
    else
        std::cout << "[HID] Opened " << device_path << ", USB=" << hid_usb_state() << std::endl;
    return hid_fd >= 0;
}

HidStatus hid_get_status() {
    return {hid_opened.load(), hid_reports.load(), hid_errors.load()};
}

std::string hid_usb_state() {
    glob_t paths{};
    std::string state = "unavailable";
    if (glob("/sys/class/udc/*/state", 0, nullptr, &paths) == 0) {
        for (size_t i = 0; i < paths.gl_pathc; ++i) {
            std::ifstream input(paths.gl_pathv[i]);
            std::string candidate;
            std::getline(input, candidate);
            if (!candidate.empty()) state = candidate;
            if (state == "configured") break;
        }
    }
    globfree(&paths);
    return state;
}

void hid_worker(float smooth_factor) {
    const float smoothing = std::max(0.01f, std::min(1.0f, smooth_factor));
    bool last_fire_state = false;
    bool initial_release = true;
    auto last_error_log = std::chrono::steady_clock::time_point::min();
    while (running) {
        if (hid_fd < 0) {
            hid_fd = open(hid_path.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
            hid_opened = hid_fd >= 0;
            if (hid_fd < 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            initial_release = true;
            last_fire_state = false;
        }

        std::unique_lock<std::mutex> lock(hid_mutex);
        if (!hid_control_enabled) { hid_buffer_x = hid_buffer_y = 0; fire_request = false; }
        hid_cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
            return !running || initial_release || std::abs(hid_buffer_x) > 0.5f ||
                   std::abs(hid_buffer_y) > 0.5f ||
                   (hid_control_enabled.load() && fire_request.load()) != last_fire_state;
        });
        if (!running) break;
        if (!initial_release && std::abs(hid_buffer_x) <= 0.5f &&
            std::abs(hid_buffer_y) <= 0.5f &&
            (hid_control_enabled.load() && fire_request.load()) == last_fire_state) continue;
        lock.unlock();
        pollfd pfd{hid_fd, POLLOUT, 0};
        if (poll(&pfd, 1, 10) <= 0) continue;
        lock.lock();

        auto step = [smoothing](float value) {
            if (std::abs(value) <= 0.5f) return 0;
            float amount = value * smoothing;
            if (std::abs(amount) < 1.0f) amount = value > 0 ? 1.0f : -1.0f;
            return static_cast<int>(std::max(-127.0f, std::min(127.0f, amount)));
        };
        const bool can_move = !initial_release && hid_control_enabled.load();
        const bool current_fire = can_move && fire_request.load();
        const int dx = can_move ? step(hid_buffer_x) : 0;
        const int dy = can_move ? step(hid_buffer_y) : 0;
        const uint8_t report[] = {static_cast<uint8_t>(current_fire ? 1 : 0),
                                 static_cast<uint8_t>(dx), static_cast<uint8_t>(dy), 0};
        const ssize_t written = write(hid_fd, report, sizeof(report));
        if (written == static_cast<ssize_t>(sizeof(report))) {
            hid_buffer_x -= dx;
            hid_buffer_y -= dy;
            last_fire_state = current_fire;
            initial_release = false;
            ++hid_reports;
        } else {
            const int error = written < 0 ? errno : EIO;
            ++hid_errors;
            const auto now = std::chrono::steady_clock::now();
            if (last_error_log == std::chrono::steady_clock::time_point::min() ||
                now - last_error_log >= std::chrono::seconds(2)) {
                std::cerr << "[HID] Report not sent: " << strerror(error)
                          << "; USB=" << hid_usb_state() << std::endl;
                last_error_log = now;
            }
            // Failed reports must not advance the button state or consume movement.
            if (error == ENODEV || error == ESHUTDOWN || error == EBADF) {
                close(hid_fd);
                hid_fd = -1;
                hid_opened = false;
                hid_buffer_x = hid_buffer_y = 0;
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }
    // Release buttons on an orderly exit; never block waiting for a disconnected host.
    if (hid_fd >= 0) {
        const uint8_t release[4] = {0, 0, 0, 0};
        pollfd pfd{hid_fd, POLLOUT, 0};
        if (poll(&pfd, 1, 20) > 0) {
            const ssize_t sent = write(hid_fd, release, sizeof(release));
            if (sent == static_cast<ssize_t>(sizeof(release))) ++hid_reports;
            else ++hid_errors;
        }
        close(hid_fd);
        hid_fd = -1;
    }
    hid_opened = false;
}
