//====== Graph Benchmark Suites ======//
//

#include <vector>
#include <string>
#include <fstream>
#include "common.h"
#include "def.h"
#include "openG.h"

using namespace std;

extern void seq_SSSP(
        uint64_t * vertexlist, 
        uint64_t * edgelist, 
        uint16_t * vproplist,
        uint16_t * eproplist,
        uint64_t vertex_cnt, 
        uint64_t edge_cnt,
        uint64_t root);
extern void parallel_SSSP(
        uint64_t * vertexlist, 
        uint64_t * edgelist, 
        uint16_t * vproplist,
        uint16_t * eproplist,
        uint64_t vertex_cnt, 
        uint64_t edge_cnt,
        uint64_t root, unsigned threadnum);

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
    arg.add_arg("root","0","root/starting vertex");
}
//==============================================================//


//==============================================================//

void output(vector<uint16_t> & vproplist)
{
    cout<<"SSSP Results:\n";
    for(size_t i=0;i<vproplist.size();i++)
    {
        cout<<"== vertex "<<i<<": level "<<vproplist[i]<<endl;
    }
}

//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: SSSP\n";

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

    size_t root,threadnum;
    arg.get_value("root",root);
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
#else
    (void)t1;
    (void)t2;
#endif

    //================================================//
    vector<uint16_t> vproplist(vertex_num, 0);
    vector<uint16_t> eproplist(edge_num, 1);
    //================================================//
    

    t1 = timer::get_usec();
    //================================================//
    if (threadnum==1)
        seq_SSSP(&(vertexlist[0]), 
            &(edgelist[0]), &(vproplist[0]),&(eproplist[0]), 
            vertexlist.size()-1, edgelist.size(), root);
    else
        parallel_SSSP(&(vertexlist[0]), 
            &(edgelist[0]), &(vproplist[0]),&(eproplist[0]), 
            vertexlist.size()-1, edgelist.size(), root, threadnum);
    //================================================//
    t2 = timer::get_usec();
    

    cout<<"\nSSSP finish: \n";
    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

#ifdef ENABLE_OUTPUT
    cout<<"\n";
    output(vproplist);
#endif

    cout<<"==================================================================\n";
    return 0;
}  // end main

