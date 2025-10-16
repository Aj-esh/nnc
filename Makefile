CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99 -Isrc
LDFLAGS = -lm

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Source files
SOURCES = $(SRC_DIR)/main.c $(SRC_DIR)/optax/net.c
OBJECTS = $(OBJ_DIR)/main.o $(OBJ_DIR)/net.o
TARGET = $(BIN_DIR)/nnc

# Default target
all: $(TARGET)

# Create directories
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build target
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
	@echo "Build successful! Executable: $(TARGET)"

# Compile main.c
$(OBJ_DIR)/main.o: $(SRC_DIR)/main.c $(SRC_DIR)/optax/net.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $(SRC_DIR)/main.c -o $(OBJ_DIR)/main.o

# Compile net.c
$(OBJ_DIR)/net.o: $(SRC_DIR)/optax/net.c $(SRC_DIR)/optax/net.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $(SRC_DIR)/optax/net.c -o $(OBJ_DIR)/net.o

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Clean complete!"

# Run the program
run: $(TARGET)
	./$(TARGET)

# Rebuild
rebuild: clean all

.PHONY: all clean run rebuild
