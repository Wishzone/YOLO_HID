#include "../src/modules/hid_mouse.h"
#include "../src/core/globals.h"
#include <cassert>
#include <cerrno>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>

static std::atomic<bool> fail_next(false);
extern "C" ssize_t __real_write(int, const void*, size_t);
extern "C" ssize_t __wrap_write(int fd, const void* data, size_t size) {
    if (fail_next.exchange(false)) { errno = EAGAIN; return -1; }
    return __real_write(fd, data, size);
}

static void read_report(int fd, uint8_t (&report)[4]) {
    pollfd pfd{fd, POLLIN, 0};
    assert(poll(&pfd, 1, 2000) > 0);
    assert(read(fd, report, sizeof(report)) == 4);
}

int main() {
    char directory[] = "/tmp/yolo-hid-test-XXXXXX";
    assert(mkdtemp(directory));
    const std::string path = std::string(directory) + "/mouse";
    assert(mkfifo(path.c_str(), 0600) == 0);
    const int reader = open(path.c_str(), O_RDONLY | O_NONBLOCK);
    assert(reader >= 0 && hid_init(path.c_str()));
    std::thread worker([] { hid_worker(1.0f); });
    uint8_t report[4];
    read_report(reader, report);
    assert(report[0] == 0 && report[1] == 0 && report[2] == 0);
    {
        std::lock_guard<std::mutex> lock(hid_mutex);
        hid_buffer_x = 300; hid_buffer_y = -260;
    }
    hid_cv.notify_one();
    int x = 0, y = 0;
    while (x != 300 || y != -260) {
        read_report(reader, report);
        x += static_cast<int8_t>(report[1]); y += static_cast<int8_t>(report[2]);
        assert(report[0] == 0 && report[3] == 0);
    }
    // A failed button-only write must retry, even with no new movement to wake it.
    fail_next = true;
    fire_request = true;
    hid_cv.notify_one();
    read_report(reader, report);
    assert(report[0] == 1);
    // Pausing must release an already pressed button and discard pending movement.
    {
        std::lock_guard<std::mutex> lock(hid_mutex);
        hid_control_enabled = false;
        hid_buffer_x = 50;
    }
    hid_cv.notify_one();
    read_report(reader, report);
    assert(report[0] == 0 && report[1] == 0 && report[2] == 0);
    assert(hid_get_status().errors == 1);
    running = false;
    hid_cv.notify_all();
    worker.join();
    read_report(reader, report);
    assert(report[0] == 0);
    close(reader); unlink(path.c_str()); rmdir(directory);
    puts("PASS HID: signed movement, report splitting, failed-write retry, pause release and shutdown");
}
