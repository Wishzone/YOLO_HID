#include "globals.h"

std::atomic<bool> running(true);
std::atomic<bool> fire_request(false);
std::atomic<int> active_connections(0);

float hid_buffer_x = 0.0f;
float hid_buffer_y = 0.0f;
std::mutex hid_mutex;
std::condition_variable hid_cv;
