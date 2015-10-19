PFM=libpfm-4.5.0
CXX=g++
CXX_FLAGS=-I./include -L./lib -O3
LIBS=-lpfm_cxx -lpfm


all: libpfm.a libpfm_cxx.a

libpfm.a: ${PFM}
	(cd ${PFM}; make -j 4; cp lib/libpfm.a ../lib/; cp -r include/perfmon ../include/)

${PFM}:
	rm -rf ${PFM}
	tar xzvf ${PFM}.tar.gz

libpfm_cxx.a: 
	make -C pfm_cxx all 

test:
	$(CXX) $(CXX_FLAGS) -c $@.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $@.o $(LIBS)

clean:
	make -C pfm_cxx clean
	rm -rf ${PFM} test *.o

