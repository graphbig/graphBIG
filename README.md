<pre style="display:inline-block;line-height:13px;">
  ________                    .__   __________.___  ________
 /  _____/___________  ______ |  |__\______   \   |/  _____/
/   \  __\_  __ \__  \ \____ \|  |  \|    |  _/   /   \  ___
\    \_\  \  | \// __ \|  |_> >   Y  \    |   \   \    \_\  \
 \______  /__|  (____  /   __/|___|  /______  /___|\______  /
        \/           \/|__|        \/       \/            \/
</pre>

# GraphBIG
GraphBIG is a graph benchmarking effort initiated by Georgia Tech __HPArch__ and inspired by IBM __System G__. By supporting a wide selection of workloads from both __CPU__ and __GPU__ sides,
GraphBIG covers the broad spectrum of graph computing and fulfills multiple major requirements, including _framework_, _representativeness_, _coverage_, and graph _data support_.


### Introduction
GraphBIG is a comprehensive benchmark suites for graph computing. The workloads are selected from
real-world use cases of IBM System G customers. GraphBIG covers a broad scope of graph computing applications,
much more than simple graph traversals.
To ensure the representativeness and coverage of the workloads, we analyzed real-world
use cases and summarized graph computing features by computation types and graph data sources.
GraphBIG workloads then cover all major computation types and data sources. 

GraphBIG benchmarks were built on an open source graph framework
named "openG", which follows the similar design methodology as IBM System G framework.
It represents architectural/system behaviors of real-world graph computing practices.

(For commercial packages of the IBM System G, please visit [IBM System G])

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

### Publication
Lifeng Nai, Yinglong Xia, Ilie G. Tanase, Hyesoon Kim, and Ching-Yung Lin. [GraphBIG: Understanding Graph Computing in the Context of Industrial Solutions](http://nailifeng.org/pubs/sc-graphbig.pdf), To appear in _the proccedings of the International Conference for High Performance Computing, Networking, Storage and Analysis(SC), Nov. 2015_

### Tutorial
[The World is Big and Linked: Whole Spectrum Industry Solutions towards Big Graphs](http://cci.drexel.edu/bigdata/bigdata2015/tutorials.html), _IEEE BigData 2015, Oct. 2015_

### Updates
v3.2 is released. It includes a few new workloads, multiple issue fixes, simulation annotations, and a new compile/test structure. Please feel to free to contact us if you notice an issue. 

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

### Documents

Documents can be found in the [GraphBIG-Doc](https://github.com/graphbig/GraphBIG-Doc) repository in the same graphbig organization. 

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
3.2

### Contact us
Lifeng Nai (lnai3 _at_ gatech.edu / nailifeng _at_ gmail.com)

Or submit issues via github

**Graph Computing, Hell Yeah!**

[IBM System G]:http://systemg.research.ibm.com/
[libpfm]:http://perfmon2.sourceforge.net/
