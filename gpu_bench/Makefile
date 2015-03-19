SUBDIR=gpu_BetweennessCentr  gpu_BFS  gpu_ConnectedComp  gpu_DegreeCentr  gpu_GraphColoring  gpu_kCore  gpu_SSSP  gpu_TriangleCount

default: all

all: 
	@for d in ${SUBDIR}; do \
          ${MAKE} -C $$d all; \
        done

clean:
	@for d in ${SUBDIR}; do \
          ${MAKE} -C $$d clean; \
        done

run: 
	@for d in ${SUBDIR}; do \
	  ${MAKE} -C $$d run; \
	done
