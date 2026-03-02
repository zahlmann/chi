CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic -D_POSIX_C_SOURCE=200809L

BIN_DIR := bin

all: $(BIN_DIR)/chi_demo

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

chi.o: chi.c chi.h
	$(CC) $(CFLAGS) -c chi.c -o chi.o

examples/demo.o: examples/demo.c chi.h
	$(CC) $(CFLAGS) -I. -c examples/demo.c -o examples/demo.o

$(BIN_DIR)/chi_demo: chi.o examples/demo.o | $(BIN_DIR)
	$(CC) $(CFLAGS) chi.o examples/demo.o -o $(BIN_DIR)/chi_demo

tests/test_chi.o: tests/test_chi.c chi.h
	$(CC) $(CFLAGS) -I. -c tests/test_chi.c -o tests/test_chi.o

$(BIN_DIR)/test_chi: chi.o tests/test_chi.o | $(BIN_DIR)
	$(CC) $(CFLAGS) chi.o tests/test_chi.o -o $(BIN_DIR)/test_chi

test: $(BIN_DIR)/test_chi
	./$(BIN_DIR)/test_chi

clean:
	rm -rf $(BIN_DIR) chi.o examples/demo.o tests/test_chi.o

.PHONY: all test clean
