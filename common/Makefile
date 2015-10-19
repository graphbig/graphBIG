ROOT=../../../
CC = g++
LINK = g++
CFLAGS = -g -I${ROOT}/apps/common/ -I${ROOT}/include/ -I${ROOT}/datastructure/graph/dbstore -I${ROOT}/datastructure/graph

all:
	$(CC) $(CFLAGS) -c Helper.cpp
	$(CC) $(CFLAGS) -c test.cpp
	$(LINK) $(CFLAGS) test.o Helper.o -o test.exe

clean:
	rm -rf *~
	rm -rf *.o
	rm -rf test.exe
