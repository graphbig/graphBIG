ROOT=../..
TARGET=betweenness
OBJS=main.o betweenness.o 
RUN_ARGS=--dataset $(ROOT)/dataset/small  

include ../common.mk

betweenness.o: betweenness.cu
	${NVCC} -c -I../cudalib $<


