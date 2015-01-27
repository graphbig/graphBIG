<pre style="display:inline-block;line-height:13px;">
  ________                    .__   __________.___  ________ 
 /  _____/___________  ______ |  |__\______   \   |/  _____/ 
/   \  __\_  __ \__  \ \____ \|  |  \|    |  _/   /   \  ___ 
\    \_\  \  | \// __ \|  |_> >   Y  \    |   \   \    \_\  \
 \______  /__|  (____  /   __/|___|  /______  /___|\______  /
        \/           \/|__|        \/       \/            \/ 
</pre>

# GraphBIG
GraphBIG is a comprehensive graph benchmarking effort inspired from IBM __System G__ and Georgia Tech __HPArch__. It covers the broad spectrum of graph computing by fulfilling multiple major requirements, including __framework__, 
__representativeness__, __coverage__, and graph __data support__.

| | | |
|:----:|---|----|
|**Graph**| => | Graphs -- large or small, static or dynamic, topological or semantic, and properties or bayesian |
|**B**| => |Benchmark Suites|
|**I**|=>|IBM System G  (http://systemg.research.ibm.com)  |
|**G**|=>|Georgia Tech HPArch (http://comparch.gatech.edu/hparch)|


### Introduction
In GraphBIG, to ensure the representativeness
and coverage of the workloads, we analyzed real-world
use cases from IBM System G customers and summarize graph computing features 
by computation types and graph data sources.
The workloads in GraphBIG are then selected from the use cases 
to cover all computation types.
In addition, GraphBIG also provides real-world data
sets covering major graph data sources and a synthetic data set
for characterization purposes. 

GraphBIG contains two branches, commercial branch and community branch.
The benchmarks of GraphBIG was initially built on the
IBM System G framework to represent real-world graph computing practices.
Later, for the open-source purpose, a community branch was established
by designing a new graph middleware, the openG framework. 
Both branches are sharing similar workloads and datasets. Their underlying frameworks
also show similar architectural behaviors. 

The __community branch__ is released here. 
For packages of the __commercial branch__, please contact [IBM System G].

### Features
GraphBIG contains the following main features
- Framework: _based on the property graph framwork from real-world graph computing practices_
- Representativeness: _workloads are selected from real-world use cases_
- Coverage: _covers multiple graph computation types, much more than just graph traversal_
- Dataset: _provides both real-world and synthetic datasets_
- C++ code base: _pure C++ code requiring only C++0x. can be supported by most gcc versions_
- Standalone package: _can be compiled without external libraries_
- Perf tools: _provides tools to profile the code section of interest with hardware performance counters ([libpfm] code is integrated)_ 



### Compile/Run

Only requirement: your gcc/g++ needs to support c++0x.

```sh
$ git clone [git-repo-url] GraphBIG
$ cd GraphBIG
$ cd benchmark
$ make clean all
$ cd [bench dir]
$ make run
$ cat output.log
```

### Authors
- Lifeng Nai, Georgia Tech (lnai3 _at_ gatech _dot_ edu / lifeng _at_ us _dot_ ibm _dot_ com)  
- Yinglong Xia, IBM T.J. Watson Research Center (yxia _at_ us _dot_ ibm _dot_ com)  
- Ilie G. Tanase, IBM T.J. Watson Research Center  
- Hyesoon Kim, Georgia Tech  
- Ching-Yung Lin, IBM T.J. Watson Research Center

### Development

Want to contribute? Great!

GraphBIG benchmarks and underlying framework are pure C++ codes with a bit STL. 
You are more than welcome to contribute new workloads, new datasets, or new tools. Please
feel free to contact us. 

### License
BSD license

### Version
1.0

**Graph Computing, Hell Yeah!**

[IBM System G]:http://systemg.research.ibm.com/
[libpfm]:http://perfmon2.sourceforge.net/
