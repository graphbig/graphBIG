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
	@rm -f output.log

run: 
	@rm -f output.log
	@for d in ${SUBDIR}; do \
	  ${MAKE} -C $$d run; \
	  cat $$d/output.log >> output.log; \
	done
