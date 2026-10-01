#ifndef GLOBALS_H
#define GLOBALS_H

#include <atomic>
#include <mutex>
#include <condition_variable>

extern std::atomic<bool> running;
extern std::atomic<bool> fire_request;
extern std::atomic<int> active_connections;
extern std::atomic<int> hid_target_class;
extern std::atomic<bool> hid_control_enabled;

extern float hid_buffer_x;
extern float hid_buffer_y;
extern std::mutex hid_mutex;
extern std::condition_variable hid_cv;

#endif // GLOBALS_H
