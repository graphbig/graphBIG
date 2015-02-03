CXX=g++
CXX_FLAGS=-g -std=c++0x
EXTRA_CXX_FLAGS=-I../

TARGET=test test2 test3

all: $(TARGET)


%: %.cpp
	$(CXX) $(CXX_FLAGS) $(EXTRA_CXX_FLAGS) $< -o $@

clean:
	@rm -rf  $(TARGET)
