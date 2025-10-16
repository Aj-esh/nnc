CC = gcc
CFLAGS = -Wall -Wextra -g -O2 -std=c99
LDFLAGS = -lm

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Source files
SOURCES = $(SRC_DIR)/la/linalg.c \
          $(SRC_DIR)/la/normal.c \
          $(SRC_DIR)/poolla/blas.c \
          $(SRC_DIR)/optax/net.c \
          main.c

# Object files
OBJECTS = $(OBJ_DIR)/linalg.o \
          $(OBJ_DIR)/normal.o \
          $(OBJ_DIR)/blas.o \
          $(OBJ_DIR)/net.o \
          $(OBJ_DIR)/main.o

# Target executable
TARGET = $(BIN_DIR)/nnc

all: directories $(TARGET)

directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

$(OBJ_DIR)/linalg.o: $(SRC_DIR)/la/linalg.c $(SRC_DIR)/la/linalg.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)/la/linalg.c -o $(OBJ_DIR)/linalg.o

$(OBJ_DIR)/normal.o: $(SRC_DIR)/la/normal.c $(SRC_DIR)/la/normal.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)/la/normal.c -o $(OBJ_DIR)/normal.o

$(OBJ_DIR)/blas.o: $(SRC_DIR)/poolla/blas.c $(SRC_DIR)/poolla/blas.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)/poolla/blas.c -o $(OBJ_DIR)/blas.o

$(OBJ_DIR)/net.o: $(SRC_DIR)/optax/net.c $(SRC_DIR)/optax/net.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)/optax/net.c -o $(OBJ_DIR)/net.o

$(OBJ_DIR)/main.o: main.c $(SRC_DIR)/optax/net.h
	$(CC) $(CFLAGS) -c main.c -o $(OBJ_DIR)/main.o

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean directories run
