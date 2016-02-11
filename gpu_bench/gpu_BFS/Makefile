ROOT=../..
TARGET=bfs_topo_frontier
OBJS=main.o bfs_topo_frontier.o 
RUN_ARGS=--dataset $(ROOT)/dataset/small --root 31 
EXTRA_TARGETS=bfs_topo_atomic bfs_topo_unroll bfs_topo_thread_centric bfs_topo_warp_centric bfs_data_thread_centric bfs_data_warp_centric

include ../common.mk

bfs_topo_atomic: main.o bfs_topo_atomic.o
	${NVCC} ${NVCC_LINK_OPTIONS} main.o bfs_topo_atomic.o -o $@ ${LIBS}

bfs_topo_unroll: main.o bfs_topo_unroll.o
	${NVCC} ${NVCC_LINK_OPTIONS} main.o bfs_topo_unroll.o -o $@ ${LIBS}

bfs_topo_thread_centric: main.o bfs_topo_thread_centric.o
	${NVCC} ${NVCC_LINK_OPTIONS} main.o bfs_topo_thread_centric.o -o $@ ${LIBS}

bfs_topo_warp_centric: main.o bfs_topo_warp_centric.o
	${NVCC} ${NVCC_LINK_OPTIONS} main.o bfs_topo_warp_centric.o -o $@ ${LIBS}

bfs_data_thread_centric: main.o bfs_data_thread_centric.o
	${NVCC} ${NVCC_LINK_OPTIONS} main.o bfs_data_thread_centric.o -o $@ ${LIBS}

bfs_data_warp_centric: main.o bfs_data_warp_centric.o
	${NVCC} ${NVCC_LINK_OPTIONS} main.o bfs_data_warp_centric.o -o $@ ${LIBS}


