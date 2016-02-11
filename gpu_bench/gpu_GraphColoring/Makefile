ROOT=../..
TARGET=gc_topo_thread_centric
OBJS=main.o gc_topo_thread_centric.o 
RUN_ARGS=--dataset $(ROOT)/dataset/small 
EXTRA_TARGETS=gc_topo_warp_centric gc_data_thread_centric gc_data_warp_centric

include ../common.mk

gc_topo_warp_centric: main.o gc_topo_warp_centric.o
	${NVCC} ${NVCC_LINK_OPTIONS} main.o gc_topo_warp_centric.o -o $@ ${LIBS}

gc_data_thread_centric: main.o gc_data_thread_centric.o
	${NVCC} ${NVCC_LINK_OPTIONS} main.o gc_data_thread_centric.o -o $@ ${LIBS}

gc_data_warp_centric: main.o gc_data_warp_centric.o
	${NVCC} ${NVCC_LINK_OPTIONS} main.o gc_data_warp_centric.o -o $@ ${LIBS}


