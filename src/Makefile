CXX = g++
CC = gcc
CFLAGS = -O3 -fPIC -I/usr/include
CXXFLAGS = -O3 -std=c++11 -I/usr/include `pkg-config --cflags opencv4`
LDFLAGS = -lrknnrt -lm -lpthread -lrga `pkg-config --libs opencv4`

TARGET_APP = ../yolo_app
TARGET_APP_26 = ../yolo_app_26

OBJS = yolo_rknn.o main.o v4l2_driver.o
OBJS_26 = yolo26_rknn.o main_yolo26.o v4l2_driver.o

all: $(TARGET_APP) $(TARGET_APP_26)

$(TARGET_APP): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS)

$(TARGET_APP_26): $(OBJS_26)
	$(CXX) -o $@ $(OBJS_26) $(LDFLAGS)

yolo_rknn.o: yolo_rknn.c
	$(CC) $(CFLAGS) -c $< -o $@

yolo26_rknn.o: yolo26_rknn.c
	$(CC) $(CFLAGS) -c $< -o $@

v4l2_driver.o: v4l2_driver.c
	$(CC) $(CFLAGS) -c $< -o $@

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

main_yolo26.o: main_yolo26.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET_APP) $(TARGET_APP_26) *.o
