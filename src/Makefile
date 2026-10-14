# Compatibility entry point. Keep all build rules in the project root.
BUILD_DIR ?= ../build/obj
TARGET_APP_26 ?= ../yolo_app_26
PROJECT_DIR := $(abspath ..)
ROOT_BUILD_DIR := $(patsubst $(PROJECT_DIR)/%,%,$(abspath $(BUILD_DIR)))
ROOT_TARGET_APP_26 := $(patsubst $(PROJECT_DIR)/%,%,$(abspath $(TARGET_APP_26)))

.PHONY: all clean rebuild package check-deps test
all clean rebuild package check-deps test:
	$(MAKE) -C .. $@ BUILD_DIR="$(ROOT_BUILD_DIR)" TARGET_APP_26="$(ROOT_TARGET_APP_26)"
