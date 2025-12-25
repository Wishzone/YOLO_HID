CC = gcc
CFLAGS = -O3 -fPIC -I/usr/include
LDFLAGS = -shared -lrknnrt -lm

TARGET = librknn_yolo.so
SRCS = yolo_rknn.c

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
