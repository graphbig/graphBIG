CXX_FLAGS+=-std=c++0x -Wall -Wno-deprecated
INCLUDE+=-I${ROOT}/common -I${ROOT}/openG
EXTRA_CXX_FLAGS+=-L${ROOT}/tools/lib

NVCC=nvcc

OUTPUT_LOG=output.log

LIBS=$(EXTRA_LIBS)

# Disable PFM for GPU workloads
CXX_FLAGS += -DNO_PFM

ifeq (${OCELOT},1)
  NVCC_FLAGS += -arch sm_20
  LIBS += -locelot
endif

ifeq (${DEBUG},1)
  CXX_FLAGS += -DDEBUG -g
else
  CXX_FLAGS +=-O3
endif

ifeq (${GSHELL},1)
  CXX_FLAGS += -DEXTERNAL_CSR
endif

ifeq (${VERIFY},1)
  CXX_FLAGS += -DENABLE_VERIFY
  NVCC_FLAGS += -DENABLE_VERIFY
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

ifeq (${OUTPUT}, 1)
  EXTRA_CXX_FLAGS+=-DENABLE_OUTPUT
  NVCC_FLAGS += -DENABLE_OUTPUT
endif

CXX_FLAGS+=$(EXTRA_CXX_FLAGS) $(INCLUDE)
LINKER_OPTIONS=$(CXX_FLAGS)
ALL_TARGETS=${TARGET} ${UNIT_TEST_TARGETS} ${EXTRA_TARGETS}

NVCC_LINK_OPTIONS+=$(EXTRA_CXX_FLAGS) $(INCLUDE)

NVCC_FLAGS+=-I${ROOT}/gpu_bench/cudalib

all: ${ALL_TARGETS}

%.o: %.cu
	${NVCC} -c ${NVCC_FLAGS} $<

.cc.o:
	${CXX} -c ${CXX_FLAGS} $<

.cpp.o:
	${CXX} -c ${CXX_FLAGS} $<


${TARGET}: ${OBJS}
	${NVCC} ${NVCC_LINK_OPTIONS} ${OBJS} -o $@ ${LIBS}

${UNIT_TEST_TARGETS}:
	${CXX} ${CXX_FLAGS} ${LIBS} -o $@ $@.cc $(LIBS)

CUB:
	@if [ -e "${ROOT}/cub" ]; then \
		echo "Linking with external CUB library"; \
	else \
		echo "Downloading CUB library from github..."; \
		git clone https://github.com/NVlabs/cub.git ${ROOT}/cub/; \
		echo "Linking with external CUB library"; \
	fi


include ${ROOT}/common.mk

