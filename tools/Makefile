PFMDIR=libpfm-4.5.0
CXX=g++
CXX_FLAGS=-I./include -L./lib -O3
LIBS=-lpfm_cxx -lpfm


all: lib/libpfm.a lib/libpfm_cxx.a

lib/libpfm.a: ${PFMDIR}
	${MAKE} -C ${PFMDIR} -j 4
	cp ${PFMDIR}/lib/libpfm.a lib/
	cp -r ${PFMDIR}/include/perfmon include/

${PFMDIR}:
	rm -rf ${PFMDIR}
	tar xzvf ${PFMDIR}.tar.gz

lib/libpfm_cxx.a: 
	make -C pfm_cxx all 

test:
	$(CXX) $(CXX_FLAGS) -c $@.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $@.o $(LIBS)

clean:
	make -C pfm_cxx clean
	rm -rf ${PFMDIR} test *.o

