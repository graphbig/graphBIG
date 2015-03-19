ROOT=../..
TARGET=connected_comp
OBJS=main.o connected_comp.o 
RUN_ARGS=--dataset $(ROOT)/dataset/small  

include ../common.mk

connected_comp.o: connected_comp.cu
	${NVCC} -c -I../cudalib $<


