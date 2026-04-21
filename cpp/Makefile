CXX      := c++
CXXFLAGS := -std=c++20 -O2 -pthread -Wall -Wextra

# Benchmark: X-TED vs reference implementation (TEST_XTED_CPU_IMPLEMENTATION)
# Requires cpp/bench_compare.cpp, cpp/ref_wrapper.{h,cpp}, and the
# reference sources under cpp/TEST_XTED_CPU_IMPLEMENTATION/ (gitignored).
BENCH_TARGET := bench_compare
BENCH_SRC    := bench_compare.cpp ref_wrapper.cpp

.PHONY: all bench run-bench clean

all: bench

bench: $(BENCH_TARGET)

$(BENCH_TARGET): $(BENCH_SRC) X-TED_C++.cpp X-TED_C++.h ref_wrapper.h \
                 TEST_XTED_CPU_IMPLEMENTATION/TED_C++.h \
                 TEST_XTED_CPU_IMPLEMENTATION/TED.cpp \
                 TEST_XTED_CPU_IMPLEMENTATION/parallel_computing.cpp \
                 TEST_XTED_CPU_IMPLEMENTATION/dataset_preprocess.cpp
	$(CXX) $(CXXFLAGS) $(BENCH_SRC) -o $(BENCH_TARGET)

run-bench: $(BENCH_TARGET)
	./$(BENCH_TARGET)

clean:
	rm -f $(BENCH_TARGET)
