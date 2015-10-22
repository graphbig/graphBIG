//====== Graph Benchmark Suites ======//
//

#include <vector>
#include <string>
#include <fstream>
#include "common.h"
#include "def.h"
#include "openG.h"

using namespace std;

extern unsigned seq_CC(
        uint64_t * vertexlist,  
        uint64_t * edgelist, uint16_t * vproplist,
        uint16_t * labellist,
        uint64_t vertex_cnt, uint64_t edge_cnt);
extern unsigned parallel_CC(
        uint64_t * vertexlist,  
        uint64_t * edgelist, uint16_t * vproplist,
        uint16_t * labellist,
        uint64_t vertex_cnt, uint64_t edge_cnt,
        unsigned threadnum);


class vertex_property
{
public:
    vertex_property():value(0){}
    vertex_property(uint64_t x):value(x){}

    uint64_t value;
};
class edge_property
{
public:
    edge_property():value(0){}
    edge_property(uint64_t x):value(x){}

    uint64_t value;
};

typedef openG::extGraph<vertex_property, edge_property> graph_t;
typedef graph_t::vertex_iterator    vertex_iterator;
typedef graph_t::edge_iterator      edge_iterator;

//==============================================================//

//==============================================================//

void output(vector<uint16_t> & labellist)
{
    cout<<"Connected Component Results:\n";
    for(size_t i=0;i<labellist.size();i++)
    {
        cout<<"== vertex "<<i<<": component "<<labellist[i]<<endl;
    }
}

//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: Connected Component\n";

    argument_parser arg;
#ifndef NO_PERF    
    gBenchPerf_event perf;
    if (arg.parse(argc,argv,perf,false)==false)
    {
        arg.help();
        return -1;
    }
#else
    if (arg.parse(argc,argv,false)==false)
    {
        arg.help();
        return -1;
    }
#endif
    string path;
    arg.get_value("dataset",path);

    size_t threadnum;
    arg.get_value("threadnum",threadnum);


    double t1, t2;
    
    cout<<"loading data... \n";

    t1 = timer::get_usec();
    string vfile = path + "/vertex.CSR";
    string efile = path + "/edge.CSR";

    vector<uint64_t> vertexlist, edgelist; 
    size_t vertex_num, edge_num;

    graph_t::load_CSR_Graph(vfile, efile,
            vertex_num, edge_num,
            vertexlist, edgelist);

    t2 = timer::get_usec();

    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
    
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

    //================================================//
    vector<uint16_t> vproplist(vertex_num, 0);
    vector<uint16_t> labellist(vertex_num, 0);
    //================================================//
    
    unsigned ret=0;
    t1 = timer::get_usec();
#ifndef NO_PERF
    perf.start();
#endif
    //================================================//
    if (threadnum==1)
        ret = seq_CC(&(vertexlist[0]), 
            &(edgelist[0]), 
            &(vproplist[0]),
            &(labellist[0]), 
            vertexlist.size()-1, 
            edgelist.size());
    else
        ret = parallel_CC(&(vertexlist[0]), 
            &(edgelist[0]), 
            &(vproplist[0]), 
            &(labellist[0]),
            vertexlist.size()-1, 
            edgelist.size(), 
            threadnum);
    //================================================//
#ifndef NO_PERF   
    perf.stop();
#endif
    t2 = timer::get_usec();
    

    cout<<"\nCC finish: \n";
    cout<<"== component: "<<ret<<endl;
    cout<<"== thread num: "<<threadnum<<endl;
    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#ifndef NO_PERF
    perf.print();
#endif
#endif

#ifdef ENABLE_OUTPUT
    cout<<"\n";
    output(labellist);
#endif

    cout<<"==================================================================\n";
    return 0;
}  // end main

