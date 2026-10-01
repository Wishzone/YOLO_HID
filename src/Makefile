CXX ?= g++
CC ?= gcc
PKG_CONFIG ?= pkg-config
CPPFLAGS += -I/usr/include $(shell $(PKG_CONFIG) --cflags opencv4)
CFLAGS ?= -O3 -fPIC
CXXFLAGS ?= -O3 -std=c++11
LDLIBS += -lrknnrt -lm -lpthread -lrga $(shell $(PKG_CONFIG) --libs opencv4)

BUILD_DIR ?= ../build/obj
TARGET_APP_26 ?= ../yolo_app_26
C_SOURCES = models/yolo26_rknn.c models/yolo26_postprocess.c modules/v4l2_driver.c
CXX_SOURCES = main_yolo26.cpp core/globals.cpp utils/sys_utils.cpp \
	modules/hid_mouse.cpp modules/web_server.cpp modules/camera_capture.cpp
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
	$(RM) $(OBJECTS) $(DEPS) $(TARGET_APP_26) ../build/test_postprocess

../build/test_postprocess: ../tests/test_postprocess.c models/yolo26_postprocess.c models/yolo26_postprocess.h core/types.h
	@mkdir -p ../build
	$(CC) $(CFLAGS) -Wall -Wextra -Werror ../tests/test_postprocess.c models/yolo26_postprocess.c -lm -o $@

test: ../build/test_postprocess
	../build/test_postprocess

-include $(DEPS)
