CXX      := g++
CXXFLAGS := -std=c++17 -O3 -Wall \
            -march=native -mtune=native \
            -flto \
            -funroll-loops \
            -fprefetch-loop-arrays

LDFLAGS  := -lm -flto


SRCS    := $(wildcard *.cpp)
OBJS    := $(SRCS:.cpp=.o)

TARGET  := main

.PHONY: all clean
all: $(TARGET)


$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)


%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

