ROOT=../..
TARGET=sssp_topo_thread_centric
OBJS=main.o sssp_topo_thread_centric.o 
RUN_ARGS=--dataset $(ROOT)/dataset/small --root 31 
EXTRA_TARGETS=sssp_topo_warp_centric sssp_data_thread_centric sssp_data_warp_centric

include ../common.mk

sssp_topo_warp_centric: main.o sssp_topo_warp_centric.o
	${NVCC} ${NVCC_LINK_OPTIONS} main.o sssp_topo_warp_centric.o -o $@ ${LIBS}

sssp_data_thread_centric: main.o sssp_data_thread_centric.o
	${NVCC} ${NVCC_LINK_OPTIONS} main.o sssp_data_thread_centric.o -o $@ ${LIBS}

sssp_data_warp_centric: main.o sssp_data_warp_centric.o
	${NVCC} ${NVCC_LINK_OPTIONS} main.o sssp_data_warp_centric.o -o $@ ${LIBS}


