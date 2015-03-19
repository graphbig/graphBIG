ROOT=../..
TARGET=kcore
OBJS=main.o kcore.o 
RUN_ARGS=--dataset $(ROOT)/dataset/small --kcore 3 

include ../common.mk

kcore.o: kcore.cu
	${NVCC} -c -I../cudalib $<


