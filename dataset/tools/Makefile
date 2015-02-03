CXX=g++
FLAGS=-O3 -std=c++11

EXECUTABLE=randgen
SOURCES=$(addsuffix .cpp, $(EXECUTABLE))

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES) 
		$(CXX) $(FLAGS) $@.cpp -o $@

run: $(SOURCES) $(EXECUTABLE)
	./randgen 1000 10000 > id.rand

clean:
	@rm -rf $(EXECUTABLE) *.o id.rand
