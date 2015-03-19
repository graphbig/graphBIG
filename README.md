<pre style="display:inline-block;line-height:13px;">
  ________                    .__   __________.___  ________
 /  _____/___________  ______ |  |__\______   \   |/  _____/
/   \  __\_  __ \__  \ \____ \|  |  \|    |  _/   /   \  ___
\    \_\  \  | \// __ \|  |_> >   Y  \    |   \   \    \_\  \
 \______  /__|  (____  /   __/|___|  /______  /___|\______  /
        \/           \/|__|        \/       \/            \/
</pre>

# GraphBIG
GraphBIG is a graph benchmarking effort initiated by IBM __System G__ and Georgia Tech __HPArch__. By supporting a wide selection of workloads from both __CPU__ and __GPU__ sides,
GraphBIG covers the broad spectrum of graph computing and fulfills multiple major requirements, including __framework__, __representativeness__, __coverage__, and graph __data support__.

| | | |
|:----:|---|----|
|__Graph__|:arrow_right:| **Graph**s -- large or small, static or dynamic, topological or semantic, and properties or bayesian |
|__B__|:arrow_right:|**B**enchmark Suites|
|__I__|:arrow_right:|**I**BM System G  (http://systemg.research.ibm.com)  |
|__G__|:arrow_right:|**G**eorgia Tech HPArch (http://comparch.gatech.edu/hparch)|

### Introduction
GraphBIG is a comprehensive benchmark suites for graph computing. The workloads are selected from
real-world use cases of IBM System G customers. GraphBIG covers a broad scope of graph computing applications,
much more than simple graph traversals.
To ensure the representativeness and coverage of the workloads, we analyzed real-world
use cases and summarized graph computing features by computation types and graph data sources.
GraphBIG workloads cover all computation types. Meanwhile, GraphBIG also provides real-world data
sets covering major graph data sources and a synthetic data set
for characterization purposes.

GraphBIG benchmarks were initially built on the
IBM System G framework to represent real-world graph computing practices.
For the open-source purpose, a community branch was established
by designing a new graph middleware, the openG framework.
Both community and commercial branch are sharing the same workloads and dataset support. Their underlying frameworks
also follow the same methodology and show the similar architectural behaviors.

The community branch is released here.
For packages of the commercial branch, please contact [IBM System G].

### Features
GraphBIG contains the following main features
- Framework: _based on the property graph framework from real-world graph computing practices_
- Representativeness: _workloads are selected from real-world use cases_
- Coverage: _covers multiple graph computation types, much more than just graph traversal_
- GPU: _provides GPU workloads under the unified framework_
- Dataset: _provides both real-world and synthetic datasets_
- C++ code base: _pure C++ code requiring only C++0x. can be supported by most gcc versions_
- Standalone package: _can be compiled without external libraries_
- Profiling tools: _provides tools to profile the code section of interest with hardware performance counters ([libpfm] code is integrated)_



### Compile/Run

- CPU benchmarks:
```sh
$ git clone https://github.com/graphbig/graphBIG.git GraphBIG
$ cd GraphBIG
$ cd benchmark
$ make clean all
$ cd [bench dir]
$ make run
$ cat output.log
```

- GPU benchmarks:
```sh
$ git clone https://github.com/graphbig/graphBIG.git GraphBIG
$ cd GraphBIG
$ cd gpu_bench
$ make clean all
$ cd [bench dir]
$ make run
$ cat output.log
```

### Datasets
To cover the diverse features of graph data, GraphBIG present two types of graph data sets, real-world data and synthetic data. The real-world data sets can illustrate real graph data features, while the synthetic data can help workload characterizations because of its flexible data size.

The detailed dataset list and download links can be found at our [wiki page](https://github.com/graphbig/graphBIG/wiki/GraphBIG-Dataset "Dataset").

### Contributors
- Lifeng Nai, _Georgia Tech_ (lnai3 _at_ gatech.edu / lifeng _at_ us.ibm.com)  
- Yinglong Xia, _IBM Thomas J. Watson Research Center_ (yxia _at_ us _dot_ ibm _dot_ com)  
- Ilie G. Tanase, _IBM Thomas J. Watson Research Center_  
- Hyesoon Kim, _Georgia Tech_  
- Ching-Yung Lin, _IBM Thomas J. Watson Research Center_

### Development

Want to contribute? Great!

GraphBIG benchmarks and underlying framework are C++ codes with a bit STL.
You are more than welcome to contribute new workloads, new datasets, or new tools. Please
feel free to contact us.

### License
BSD license

### Version
2.0

### Upcoming
__GraphBIG-GPU__ now has been released!

The profiling tools of GPU side will be released soon.

### Contact us
Lifeng Nai (lnai3 _at_ gatech.edu)

**Graph Computing, Hell Yeah!**

[IBM System G]:http://systemg.research.ibm.com/
[libpfm]:http://perfmon2.sourceforge.net/
