CXX = g++
CC = gcc
CFLAGS = -O3 -fPIC -I/usr/include
CXXFLAGS = -O3 -std=c++11 -I/usr/include `pkg-config --cflags opencv4`
LDFLAGS = -lrknnrt -lm -lpthread -lrga `pkg-config --libs opencv4`

TARGET_APP_26 = ../yolo_app_26

COMMON_OBJS = core/globals.o utils/sys_utils.o modules/hid_mouse.o modules/web_server.o modules/camera_capture.o modules/v4l2_driver.o
OBJS_26 = models/yolo26_rknn.o main_yolo26.o $(COMMON_OBJS)

all: $(TARGET_APP_26)

$(TARGET_APP_26): $(OBJS_26)
	$(CXX) -o $@ $(OBJS_26) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET_APP_26) $(OBJS_26)
