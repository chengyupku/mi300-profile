# Makefile for compiling and running fp16_tensor_core_tflops with hipcc

# Compiler and flags
HIPCC = hipcc
CXXFLAGS = -std=c++17 --offload-arch=gfx942

# Target
TARGET = fp16_tensor_core_tflops
SRC = fp16_tensor_core_tflops.cpp

# Default rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(CXXFLAGS) -o $@ $^

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
