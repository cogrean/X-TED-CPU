CXX      := c++
CXXFLAGS := -std=c++20 -O2 -pthread -Wall -Wextra
TARGET   := test_xted
SRC      := test_xted.cpp

IMPL_DIR  := TEST_XTED_CPU_IMPLEMENTATION
BENCH_BIN := bench_compare
BENCH_SRC := bench_compare.cpp
REF_OBJ   := ref_wrapper.o
REF_SRC   := ref_wrapper.cpp

.PHONY: all test bench run_bench clean

all: $(TARGET)

$(TARGET): $(SRC) X-TED_C++.cpp
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

test: $(TARGET)
	./$(TARGET)

# Compile the reference-implementation wrapper into its own object file.
# -I$(IMPL_DIR) lets the .cpp files find TED_C++.h and each other.
$(REF_OBJ): $(REF_SRC) ref_wrapper.h \
            $(IMPL_DIR)/TED_C++.h \
            $(IMPL_DIR)/TED.cpp \
            $(IMPL_DIR)/dataset_preprocess.cpp \
            $(IMPL_DIR)/parallel_computing.cpp
	$(CXX) $(CXXFLAGS) -I$(IMPL_DIR) -c $(REF_SRC) -o $(REF_OBJ)

# Link the benchmark executable.
$(BENCH_BIN): $(BENCH_SRC) $(REF_OBJ) X-TED_C++.cpp ref_wrapper.h
	$(CXX) $(CXXFLAGS) -I$(IMPL_DIR) $(BENCH_SRC) $(REF_OBJ) -o $(BENCH_BIN)

bench: $(BENCH_BIN)

run_bench: $(BENCH_BIN)
	./$(BENCH_BIN)

clean:
	rm -f $(TARGET) $(BENCH_BIN) $(REF_OBJ)
