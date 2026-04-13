CXX      := c++
CXXFLAGS := -std=c++20 -O2 -pthread -Wall -Wextra
TARGET   := test_xted
SRC      := test_xted.cpp

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(SRC) X-TED_C++.cpp
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
