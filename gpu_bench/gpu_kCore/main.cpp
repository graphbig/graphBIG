//====== Graph Benchmark Suites ======//
//

#include <vector>
#include <string>
#include <fstream>
#include "../lib/common.h"
#include "../lib/def.h"
#include "openG.h"

using namespace std;

extern unsigned cuda_kcore(
        uint64_t * vertexlist, uint64_t * degreelist, 
        uint64_t * edgelist, uint32_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt,
        unsigned kcore);

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

struct arg_t
{
    string dataset_path;
    unsigned kcore;
};

void arg_init(arg_t& arguments)
{
    arguments.dataset_path.clear();
    arguments.kcore=3;
}

void arg_parser(arg_t& arguments, vector<string>& inputarg)
{
    for (size_t i=1;i<inputarg.size();i++) 
    {

        if (inputarg[i]=="--dataset") 
        {
            i++;
            arguments.dataset_path=inputarg[i];
        }
        else if (inputarg[i]=="--kcore")
        {
            i++;
            arguments.kcore=atol(inputarg[i].c_str());
        }
        else
        {
            cerr<<"wrong argument: "<<inputarg[i]<<endl;
            return;
        }
    }
    return;
}
//==============================================================//


//==============================================================//

void output(vector<uint32_t> & vproplist, unsigned kcore)
{
    cout<<"kCore Results:\n";
    for(size_t i=0;i<vproplist.size();i++)
    {
        cout<<"== vertex "<<i<<": degree-"<<vproplist[i];
        if (vproplist[i]<kcore)
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

    arg_t arguments;
    vector<string> inputarg;
    argument_parser::initialize(argc,argv,inputarg);
    arg_init(arguments);
    arg_parser(arguments,inputarg);

    graph_t g;
    double t1, t2;
    
    cout<<"loading data... \n";

    t1 = timer::get_usec();
    string vfile = arguments.dataset_path + "/vertex.csv";
    string efile = arguments.dataset_path + "/edge.csv";

    if (g.load_csv_vertices(vfile, true, "|,", 0) == -1)
        return -1;
    if (g.load_csv_edges(efile, true, "|,", 0, 1) == -1) 
        return -1;

    size_t vertex_num = g.num_vertices();
    size_t edge_num = g.num_edges();
    t2 = timer::get_usec();

    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
    
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

    t1 = timer::get_usec();
    //================================================//
    // prepare compact data for CUDA side
    vector<uint64_t> vertexlist, degreelist, edgelist; 
    g.to_CSR_Graph(vertexlist, degreelist, edgelist);
    t2 = timer::get_usec();

    cout<<"== data conversion time: "<<t2-t1<<" sec\n"<<endl;

    vector<uint32_t> vproplist(vertex_num, 0);
    vector<bool> rmlist(vertex_num, false);
    //================================================//
    
    unsigned remove_cnt;
    t1 = timer::get_usec();
    //================================================//
    // call CUDA function 
    remove_cnt = cuda_kcore(&(vertexlist[0]), &(degreelist[0]), 
            &(edgelist[0]), &(vproplist[0]), 
            vertexlist.size(), edgelist.size(),
            arguments.kcore);
    //================================================//
    t2 = timer::get_usec();
    

    cout<<"\nGPU kCore finish: \n";
    cout<<"== kcore: "<<arguments.kcore<<"\n";
    cout<<"== remove #: "<<remove_cnt<<"\n";
    cout<<"== "<<g.num_vertices()<<" vertices  "<<g.num_edges()<<" edges\n";
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

#ifdef ENABLE_OUTPUT
    cout<<"\n";
    output(vproplist, arguments.kcore);
#endif

    cout<<"==================================================================\n";
    return 0;
}  // end main

