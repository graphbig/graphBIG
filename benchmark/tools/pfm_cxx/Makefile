CXX=g++
CXX_FLAGS=-I../include -L../lib -O3
LIBNAME=libpfm_cxx.a

all: pfm_cxx

pfm_cxx: 
	$(CXX) $(CXX_FLAGS) -c $@.cpp -lpfm
	ar -cr $(LIBNAME) *.o
	cp $(LIBNAME) ../lib
	cp pfm_cxx.h ../include

clean:
	rm -rf $(LIBNAME) *.o

.PHONY: clean

