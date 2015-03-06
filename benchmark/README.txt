=======================================
To compile all benchmarks:

$ make clean all
$ cd [bench dir]
$ make run
$ cat output.log
=======================================
1.  More compiling flags can be found 
    in check common.mk
2.  Example of benchmark arguments can 
    be found at RUN_ARGS of each Makefile
========================================
[Directory]

[bench_***]     benchmark
[ubench_***]    microbenchmark
[lib]           library header files
[tools]         profiling tool codes
[../dataset]    data sets
[../openG]      graph library files
========================================
[Version] 1.2
========================================
[UPCOMING]

GPU benchmarks will be released in
April 2015.
========================================

