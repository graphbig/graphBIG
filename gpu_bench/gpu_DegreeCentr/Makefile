ROOT=../..
TARGET=degree_centr
OBJS=main.o degree_centr.o 
RUN_ARGS=--dataset $(ROOT)/dataset/small  

include ../common.mk

degree_centr.o: degree_centr.cu
	${NVCC} -c -I../cudalib $<


