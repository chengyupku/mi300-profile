# Compiler and flags
HIPCC = hipcc
CXXFLAGS = -std=c++17 --offload-arch=gfx942

# Directories
BUILD_DIR = build
TARGET = $(BUILD_DIR)/fp16_tensor_core_tflops
SRC = fp16_flops.cpp

# Default rule
all: $(TARGET)

# Ensure build directory exists and build the target
$(TARGET): $(SRC)
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) -o $@ $^

# Run the compiled binary
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
