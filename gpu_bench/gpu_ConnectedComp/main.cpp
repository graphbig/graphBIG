//====== Graph Benchmark Suites ======//
//

#include <vector>
#include <map>
#include <string>
#include <fstream>
#include "common.h"
#include "def.h"
#include "openG.h"

using namespace std;

extern void cuda_connected_comp(
        uint64_t * vertexlist, 
        uint64_t * edgelist, uint64_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt);

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

void output(vector<uint64_t> & vproplist)
{
    map<uint64_t,uint32_t> prop2id;
    uint32_t id_gen = 0;
    cout<<"Connected Component Results:\n";
    for(size_t i=0;i<vproplist.size();i++)
    {
        if (prop2id.find(vproplist[i]) == prop2id.end())
            prop2id[vproplist[i]] = id_gen++;

        cout<<"== vertex "<<i<<": component "<<prop2id[vproplist[i]]<<endl;
    }
}

//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: GPU Connected Component\n";

    argument_parser arg;
    gBenchPerf_event perf;
    if (arg.parse(argc,argv,perf,false)==false)
    {
        arg.help();
        return -1;
    }
    string path;
    arg.get_value("dataset",path);

    double t1, t2;

    cout<<"loading data... \n";

    t1 = timer::get_usec();
    size_t vertex_num, edge_num;
#ifdef EXTERNAL_CSR
    if (g.load_CSR_Graph(path, 
            vertex_num,edge_num,vertexlist,edgelist)==false)
    {
        cout<<"cannot open csr files"<<endl;
        return -1;
    }
#else
    string vfile = path + "/vertex.CSR";
    string efile = path + "/edge.CSR";

    vector<uint64_t> vertexlist, edgelist; 
    

    graph_t::load_CSR_Graph(vfile, efile,
            vertex_num, edge_num,
            vertexlist, edgelist);
#endif    
    t2 = timer::get_usec();

    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
    
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#else
    (void)t1;
    (void)t2;
#endif


    //================================================//
    vector<uint64_t> vproplist(vertex_num, 0);
    //================================================//
    
    t1 = timer::get_usec();
    //================================================//
    // call CUDA function 
    cuda_connected_comp(&(vertexlist[0]), 
            &(edgelist[0]), &(vproplist[0]), 
            vertexlist.size()-1, edgelist.size());
    //================================================//
    t2 = timer::get_usec();
    

    cout<<"\nGPU Connected Component finish: \n";
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

