ROOT=../..
TARGET=triangle_count
OBJS=main.o triangle_count.o 
RUN_ARGS=--dataset $(ROOT)/dataset/small  

include ../common.mk

triangle_count.o: triangle_count.cu
	${NVCC} -c -I../cudalib $<


