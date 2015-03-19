CXX_FLAGS+=-std=c++0x -Wall -Wno-deprecated
INCLUDE+=-I${ROOT}/benchmark/tools/include -I${ROOT}/openG
EXTRA_CXX_FLAGS+=-L${ROOT}/benchmark/tools/lib

NVCC=nvcc

OUTPUT_LOG=output.log

LIBS=$(EXTRA_LIBS)

ifeq (${DEBUG},1)
  CXX_FLAGS += -DDEBUG -g
else
  CXX_FLAGS +=-O3
endif

ifeq (${VERIFY},1)
  CXX_FLAGS += -DENABLE_VERIFY
endif

ifeq (${STRUCTURE}, LL)
  TRAITS=-DTRAITS_LL
endif

ifeq (${STRUCTURE}, VL)
  TRAITS=-DTRAITS_VL
endif

ifeq (${STRUCTURE}, LV)
  TRAITS=-DTRAITS_LV
endif

ifeq (${STRUCTURE}, VV)
  TRAITS=-DTRAITS_VV
endif

EXTRA_CXX_FLAGS+=${TRAITS}

ifeq (${OUTPUT}, on)
  EXTRA_CXX_FLAGS+=-DENABLE_OUTPUT
endif

CXX_FLAGS+=$(EXTRA_CXX_FLAGS) $(INCLUDE)
LINKER_OPTIONS=$(CXX_FLAGS)
ALL_TARGETS=${TARGET} ${UNIT_TEST_TARGETS} ${EXTRA_TARGETS}

NVCC_LINK_OPTIONS+=$(EXTRA_CXX_FLAGS) $(INCLUDE)

all: ${ALL_TARGETS}

.cc.o:
	${CXX} -c ${CXX_FLAGS} $<

.cpp.o:
	${CXX} -c ${CXX_FLAGS} $<

${TARGET}: ${OBJS}
	${NVCC} ${NVCC_LINK_OPTIONS} ${OBJS} -o $@ ${LIBS}

${UNIT_TEST_TARGETS}:
	${CXX} ${CXX_FLAGS} ${LIBS} -o $@ $@.cc $(LIBS)

reset_generated_dir:
	@if [ -n "${GENERATED_DIRS}" ]; then \
          rm -rf ${GENERATED_DIRS}; \
          mkdir ${GENERATED_DIRS};  \
        fi

run: ${TARGET} reset_generated_dir
	@if [ -n "${TARGET}" ]; then \
          echo "Running ${TARGET}, output in ${OUTPUT_LOG}"; \
          ./${TARGET} ${RUN_ARGS} > ${OUTPUT_LOG} 2>&1; \
	fi

CUB:
	@if [ -e "${ROOT}/cub" ]; then \
		echo "Linking with external CUB library"; \
	else \
		echo "Downloading CUB library from github..."; \
		git clone https://github.com/NVlabs/cub.git ${ROOT}/cub/; \
		echo "Linking with external CUB library"; \
	fi

clean:
	@-/bin/rm -rf ${ALL_TARGETS} ${GENERATED_DIRS} *.o *~ core core.* ${OUTPUT_LOG}
