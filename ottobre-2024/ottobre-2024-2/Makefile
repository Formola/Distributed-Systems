# Makefile for MPI program
CC = mpicc
CFLAGS = -Wall -g
LDFLAGS = -lm
TARGET = main
SOURCE = main.c

# Default target
$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)

# Clean target
clean:
	rm -f $(TARGET)

# Run target (optional - runs with 9 processes)
run: $(TARGET)
	mpirun -np 3 ./$(TARGET)

# Phony targets
.PHONY: clean run