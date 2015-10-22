//====== Graph Benchmark Suites ======//
//

#include <vector>
#include <string>
#include <fstream>
#include "common.h"
#include "def.h"
#include "openG.h"

using namespace std;

extern unsigned seq_kcore(
        uint64_t * vertexlist, 
        uint64_t * edgelist, int16_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt,
        unsigned kcore);
extern unsigned parallel_kcore(
        uint64_t * vertexlist, 
        uint64_t * edgelist, int16_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt,
        unsigned kcore, unsigned threadnum);

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
void arg_init(argument_parser & arg)
{
    arg.add_arg("kcore","3","kCore k value");
}
//==============================================================//


//==============================================================//

void output(vector<int16_t> & vproplist, unsigned kcore)
{
    cout<<"kCore Results:\n";
    for(size_t i=0;i<vproplist.size();i++)
    {
        cout<<"== vertex "<<i<<": degree-"<<vproplist[i];
        if (vproplist[i]<(int16_t)kcore)
            cout<<" removed-true\n";
        else
            cout<<" removed-false\n";
    }
}

//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: GPU kCore Decomposition\n";

    argument_parser arg;
    arg_init(arg);
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

    size_t kcore,threadnum;
    arg.get_value("kcore",kcore);
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
    vector<int16_t> vproplist(vertex_num, 0);
    vector<bool> rmlist(vertex_num, false);
    //================================================//
    
    unsigned remove_cnt;
    t1 = timer::get_usec();
    //================================================//

    if (threadnum==1)
        remove_cnt = seq_kcore(&(vertexlist[0]), 
            &(edgelist[0]), &(vproplist[0]), 
            vertexlist.size()-1, edgelist.size(),
            kcore);
    else   
        remove_cnt = parallel_kcore(&(vertexlist[0]), 
            &(edgelist[0]), &(vproplist[0]), 
            vertexlist.size()-1, edgelist.size(),
            kcore, threadnum);
    //================================================//
    t2 = timer::get_usec();
    

    cout<<"\nGPU kCore finish: \n";
    cout<<"== kcore: "<<kcore<<"\n";
    cout<<"== remove #: "<<remove_cnt<<"\n";
    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

#ifdef ENABLE_OUTPUT
    cout<<"\n";
    output(vproplist, kcore);
#endif

    cout<<"==================================================================\n";
    return 0;
}  // end main

