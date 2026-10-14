CXX ?= g++
CC ?= gcc
PKG_CONFIG ?= pkg-config
PYTHON ?= python3
CPPFLAGS += -I/usr/include $(shell $(PKG_CONFIG) --cflags opencv4)
CFLAGS ?= -O3 -fPIC
CXXFLAGS ?= -O3 -std=c++11
LDLIBS += -lrknnrt -lm -lpthread -lrga $(shell $(PKG_CONFIG) --libs opencv4)

BUILD_DIR ?= build/obj
TARGET_APP_26 ?= yolo_app_26
C_SOURCES = models/yolo26_rknn.c models/yolo26_postprocess.c modules/v4l2_driver.c
CXX_SOURCES = main_yolo26.cpp core/globals.cpp utils/sys_utils.cpp \
	modules/hid_mouse.cpp modules/web_server.cpp modules/camera_capture.cpp modules/target_tracker.cpp
OBJECTS = $(addprefix $(BUILD_DIR)/,$(C_SOURCES:.c=.o) $(CXX_SOURCES:.cpp=.o))
DEPS = $(OBJECTS:.o=.d)
TEST_BINS = build/test_postprocess build/test_hid build/test_web build/test_tracker

.PHONY: all clean rebuild package check-deps test
all: check-deps $(TARGET_APP_26)

check-deps:
	@command -v $(CC) >/dev/null
	@command -v $(CXX) >/dev/null
	@$(PKG_CONFIG) --exists opencv4 || { echo 'Missing OpenCV development package (opencv4.pc)' >&2; exit 1; }

$(TARGET_APP_26): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJECTS) $(LDLIBS)

$(BUILD_DIR)/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

clean:
	$(RM) $(OBJECTS) $(DEPS) $(TARGET_APP_26) $(TEST_BINS)

build/test_postprocess: tests/test_postprocess.c src/models/yolo26_postprocess.c src/models/yolo26_postprocess.h src/core/types.h
	@mkdir -p build
	$(CC) $(CFLAGS) -Wall -Wextra -Werror tests/test_postprocess.c src/models/yolo26_postprocess.c -lm -o $@

build/test_hid: tests/test_hid.cpp src/modules/hid_mouse.cpp src/core/globals.cpp src/modules/hid_mouse.h src/core/globals.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -Wall -Wextra -Werror tests/test_hid.cpp src/modules/hid_mouse.cpp src/core/globals.cpp -lpthread -Wl,--wrap=write -o $@

build/test_web: tests/test_web.cpp src/modules/web_server.cpp src/modules/hid_mouse.cpp src/modules/target_tracker.cpp src/core/globals.cpp src/utils/sys_utils.cpp $(BUILD_DIR)/modules/v4l2_driver.o src/modules/web_server.h src/modules/target_tracker.h src/modules/v4l2_driver.h src/modules/hid_mouse.h src/core/globals.h src/core/types.h src/utils/sys_utils.h
	@mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -Wall -Wextra -Werror tests/test_web.cpp src/modules/web_server.cpp src/modules/hid_mouse.cpp src/modules/target_tracker.cpp src/core/globals.cpp src/utils/sys_utils.cpp $(BUILD_DIR)/modules/v4l2_driver.o -lrga -lpthread $(shell $(PKG_CONFIG) --libs opencv4) -Wl,--wrap=v4l2_manual_qbuf -o $@

build/test_tracker: tests/test_tracker.cpp src/modules/target_tracker.cpp src/modules/target_tracker.h src/core/types.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -Wall -Wextra -Werror tests/test_tracker.cpp src/modules/target_tracker.cpp -o $@

test: $(TEST_BINS)
	build/test_postprocess
	build/test_hid
	build/test_web
	build/test_tracker
	$(PYTHON) tests/test_edid.py

rebuild:
	$(MAKE) clean
	$(MAKE) all

package: all
	bash scripts/package.sh

-include $(DEPS)
