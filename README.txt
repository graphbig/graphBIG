  ________                    .__   __________.___  ________ 
 /  _____/___________  ______ |  |__\______   \   |/  _____/ 
/   \  __\_  __ \__  \ \____ \|  |  \|    |  _/   /   \  ___ 
\    \_\  \  | \// __ \|  |_> >   Y  \    |   \   \    \_\  \
 \______  /__|  (____  /   __/|___|  /______  /___|\______  /
        \/           \/|__|        \/       \/            \/ 

===============================================================
[GraphBIG]

GraphBIG is a comprehensive graph benchmarking effort 
initiated by IBM System G and Georgia Tech HPArch. 
It covers the broad spectrum of graph computing 
by fulfilling multiple major requirements, 
including framework, representativeness, coverage, 
and graph data support.
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
<benchmark>     benchmark workloads
===============================================================
We are working on the open-soucing of parallel/concurrent 
benchmarks of commercial branch. They'll be released soon.

GraphBIG-GPU benchmarks will also be released in near future.
===============================================================

