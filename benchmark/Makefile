SUBDIR=bench_BFS bench_DFS bench_connectedComp bench_degreeCentr bench_betweennessCentr bench_kCore bench_shortestPath bench_triangleCount bench_graphUpdate bench_graphConstruct bench_Moralization ubench_add ubench_find ubench_delete ubench_traverse

default: all

all: pfm_cxx 
	@for d in ${SUBDIR}; do \
          ${MAKE} -C $$d all; \
        done

pfm_cxx:
	${MAKE} -C tools all

clean:
	@for d in ${SUBDIR}; do \
          ${MAKE} -C $$d clean; \
        done
	${MAKE} -C tools clean

run: 
	@for d in ${SUBDIR}; do \
	  ${MAKE} -C $$d run; \
	done
