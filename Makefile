CC = gcc
CFLAGS = -Wall -Wextra -O2 -I./include -pthread
LDFLAGS = -lm -pthread

SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

SRCS = $(SRC_DIR)/main.c \
       $(SRC_DIR)/nn.c \
       $(SRC_DIR)/train.c \
       $(SRC_DIR)/val.c \
       $(SRC_DIR)/data.c \
       $(SRC_DIR)/act.c \
       $(SRC_DIR)/optax.c \
       $(SRC_DIR)/la/linalg.c \
       $(SRC_DIR)/la/normal.c \
       $(SRC_DIR)/poolla/blas.c \
       $(SRC_DIR)/poolla/thread_pool.c

OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

TARGET = $(BUILD_DIR)/nnc

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET)
