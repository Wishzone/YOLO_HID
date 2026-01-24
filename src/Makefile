CXX = g++
CC = gcc
CFLAGS = -O3 -fPIC -I/usr/include
CXXFLAGS = -O3 -std=c++11 -I/usr/include `pkg-config --cflags opencv4`
LDFLAGS = -lrknnrt -lm -lpthread -lrga `pkg-config --libs opencv4`

TARGET_APP = ../yolo_app

OBJS = yolo_rknn.o main.o v4l2_driver.o

all: $(TARGET_APP)

$(TARGET_APP): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS)

yolo_rknn.o: yolo_rknn.c
	$(CC) $(CFLAGS) -c $< -o $@

v4l2_driver.o: v4l2_driver.c
	$(CC) $(CFLAGS) -c $< -o $@

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET_APP) *.o
