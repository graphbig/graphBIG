  ________                    .__   __________.___  ________ 
 /  _____/___________  ______ |  |__\______   \   |/  _____/ 
/   \  __\_  __ \__  \ \____ \|  |  \|    |  _/   /   \  ___ 
\    \_\  \  | \// __ \|  |_> >   Y  \    |   \   \    \_\  \
 \______  /__|  (____  /   __/|___|  /______  /___|\______  /
        \/           \/|__|        \/       \/            \/ 

===============================================================
[GraphBIG]

GraphBIG is a graph benchmarking effort initiated by 
IBM System G and Georgia Tech HPArch. By supporting a 
wide selection of workloads from both CPU and GPU sides, 
GraphBIG covers the broad spectrum of graph computing 
and fulfills multiple major requirements, including framework, 
representativeness, coverage, and graph data support.
===============================================================
[Compile/Run]

$ git clone [git-repo-url] GraphBIG
$ cd GraphBIG
$ cd benchmark
$ make clean all
$ cd [bench dir]
$ make run
$ cat output.log
===============================================================
[Directory]

<dataset>       data set files
<openG>         graph library
<benchmark>     workloads
===============================================================
[Workloads]

CPU side:
11 benchmarks & 4 micro-benchmarks

GPU side:
8 GPU workloads
===============================================================
[Dataset]

A small size test data is included with the source code. Please 
check the wiki pages of our github repo for the download links
of other datasets. 
===============================================================

