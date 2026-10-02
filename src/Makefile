CXX ?= g++
CC ?= gcc
PKG_CONFIG ?= pkg-config
PYTHON ?= python3
CPPFLAGS += -I/usr/include $(shell $(PKG_CONFIG) --cflags opencv4)
CFLAGS ?= -O3 -fPIC
CXXFLAGS ?= -O3 -std=c++11
LDLIBS += -lrknnrt -lm -lpthread -lrga $(shell $(PKG_CONFIG) --libs opencv4)

BUILD_DIR ?= ../build/obj
TARGET_APP_26 ?= ../yolo_app_26
C_SOURCES = models/yolo26_rknn.c models/yolo26_postprocess.c modules/v4l2_driver.c
CXX_SOURCES = main_yolo26.cpp core/globals.cpp utils/sys_utils.cpp \
	modules/hid_mouse.cpp modules/web_server.cpp modules/camera_capture.cpp modules/target_tracker.cpp
OBJECTS = $(addprefix $(BUILD_DIR)/,$(C_SOURCES:.c=.o) $(CXX_SOURCES:.cpp=.o))
DEPS = $(OBJECTS:.o=.d)

.PHONY: all clean check-deps test
all: check-deps $(TARGET_APP_26)

check-deps:
	@command -v $(CC) >/dev/null
	@command -v $(CXX) >/dev/null
	@$(PKG_CONFIG) --exists opencv4 || { echo 'Missing OpenCV development package (opencv4.pc)' >&2; exit 1; }

$(TARGET_APP_26): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJECTS) $(LDLIBS)

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

clean:
	$(RM) $(OBJECTS) $(DEPS) $(TARGET_APP_26) ../build/test_postprocess ../build/test_hid ../build/test_web ../build/test_tracker

../build/test_postprocess: ../tests/test_postprocess.c models/yolo26_postprocess.c models/yolo26_postprocess.h core/types.h
	@mkdir -p ../build
	$(CC) $(CFLAGS) -Wall -Wextra -Werror ../tests/test_postprocess.c models/yolo26_postprocess.c -lm -o $@

../build/test_hid: ../tests/test_hid.cpp modules/hid_mouse.cpp core/globals.cpp modules/hid_mouse.h core/globals.h
	@mkdir -p ../build
	$(CXX) $(CXXFLAGS) -Wall -Wextra -Werror ../tests/test_hid.cpp modules/hid_mouse.cpp core/globals.cpp -lpthread -Wl,--wrap=write -o $@

../build/test_web: ../tests/test_web.cpp modules/web_server.cpp modules/hid_mouse.cpp modules/target_tracker.cpp core/globals.cpp utils/sys_utils.cpp $(BUILD_DIR)/modules/v4l2_driver.o modules/web_server.h modules/target_tracker.h modules/v4l2_driver.h modules/hid_mouse.h core/globals.h core/types.h utils/sys_utils.h
	@mkdir -p ../build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -Wall -Wextra -Werror ../tests/test_web.cpp modules/web_server.cpp modules/hid_mouse.cpp modules/target_tracker.cpp core/globals.cpp utils/sys_utils.cpp $(BUILD_DIR)/modules/v4l2_driver.o -lrga -lpthread $(shell $(PKG_CONFIG) --libs opencv4) -Wl,--wrap=v4l2_manual_qbuf -o $@

../build/test_tracker: ../tests/test_tracker.cpp modules/target_tracker.cpp modules/target_tracker.h core/types.h
	@mkdir -p ../build
	$(CXX) $(CXXFLAGS) -Wall -Wextra -Werror ../tests/test_tracker.cpp modules/target_tracker.cpp -o $@

test: ../build/test_postprocess ../build/test_hid ../build/test_web ../build/test_tracker
	../build/test_postprocess
	../build/test_hid
	../build/test_web
	../build/test_tracker
	$(PYTHON) ../tests/test_edid.py

-include $(DEPS)
