SUBDIR=csr_BFS csr_DC csr_CComp csr_GraphColoring csr_kCore csr_SSSP csr_TC
SUBDIR+=tool_genCSR

default: all

all: pfm_cxx 
	@for d in ${SUBDIR}; do \
          ${MAKE} -C $$d all; \
        done

pfm_cxx:
	${MAKE} -C ../tools all

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

