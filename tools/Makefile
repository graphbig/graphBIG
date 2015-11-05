PFMDIR=libpfm-4.5.0
CXX=g++
CXX_FLAGS=-I./include -L./lib -O3
LIBS=-lpfm_cxx -lpfm


all: libpfm.a libpfm_cxx.a

libpfm.a: ${PFMDIR}
	(cd ${PFMDIR}; make -j 4; cp lib/libpfm.a ../lib/; cp -r include/perfmon ../include/)

${PFMDIR}:
	rm -rf ${PFMDIR}
	tar xzvf ${PFMDIR}.tar.gz

libpfm_cxx.a: 
	make -C pfm_cxx all 

test:
	$(CXX) $(CXX_FLAGS) -c $@.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $@.o $(LIBS)

clean:
	make -C pfm_cxx clean
	rm -rf ${PFMDIR} test *.o

