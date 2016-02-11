CXX_FLAGS+=-std=c++0x -Wall -Wno-deprecated
INCLUDE+=-I${ROOT}/common -I${ROOT}/openG
EXTRA_CXX_FLAGS+=-L${ROOT}/tools/lib

LIBS=${EXTRA_LIBS}

ifeq (${PFM},0)
  CXX_FLAGS += -DNO_PFM
else
  EXTRA_LIBS += -lpfm_cxx -lpfm
  INCLUDE += -I${ROOT}/tools/include
endif

ifeq (${DEBUG},1)
  CXX_FLAGS += -DDEBUG -g -O0
else
  CXX_FLAGS +=-O3
endif

ifeq (${OMP},0)
# do nothing
else
	CXX_FLAGS += -DUSE_OMP
endif

ifeq (${HMC},1)
  OBJS += HMC.o
  CXX_FLAGS += -DHMC
  SIM=1
endif

ifeq (${SIM},1)
  OBJS += SIM.o
  CXX_FLAGS += -DSIM
endif

ifeq (${VERIFY},1)
  CXX_FLAGS += -DENABLE_VERIFY
endif

ifeq (${EDGES}, 1)
  CXX_FLAGS += -DEDGES_ONLY
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

ifeq (${STRUCTURE}, LLS)
  TRAITS=-DTRAITS_LL_S
endif

EXTRA_CXX_FLAGS+=${TRAITS}

ifeq (${OUTPUT},1)
  EXTRA_CXX_FLAGS+=-DENABLE_OUTPUT
endif

CXX_FLAGS+=$(EXTRA_CXX_FLAGS) $(INCLUDE)
LINKER_OPTIONS+=$(CXX_FLAGS)
ALL_TARGETS=${TARGET} ${UNIT_TEST_TARGETS}

all: ${ALL_TARGETS}

.cc.o:
	${CXX} -c ${CXX_FLAGS} $<

.cpp.o:
	${CXX} -c ${CXX_FLAGS} $<

${TARGET}: ${OBJS}
	${CXX} ${LINKER_OPTIONS} ${OBJS} -o $@ ${LIBS}

${UNIT_TEST_TARGETS}:
	${CXX} ${CXX_FLAGS} ${LIBS} -o $@ $@.cc $(LIBS)

HMC.o:
	${CXX} -DUSE_OMP -c ${ROOT}/common/HMC.cpp

SIM.o:
	${CXX} -c ${ROOT}/common/SIM.cpp


include ${ROOT}/common.mk

