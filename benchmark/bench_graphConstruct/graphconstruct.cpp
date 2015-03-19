//====== Graph Benchmark Suites ======//
//======= RandomGraph Construction =======//
//
// Usage: ./randomgraph --vertex <vertex #> --edge <edge #>

#include "../lib/common.h"
#include "../lib/def.h"
#include "../lib/perf.h"
#include "openG.h"

using namespace std;

#define SEED 111

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
    size_t vertex_num;
    size_t edge_num;
};

void arg_init(arg_t& arguments)
{
    arguments.vertex_num = 100;
    arguments.edge_num = 1000;
}

void arg_parser(arg_t& arguments, vector<string>& inputarg)
{
    for (size_t i=1;i<inputarg.size();i++) 
    {

        if (inputarg[i]=="--vertex") 
        {
            i++;
            arguments.vertex_num=atol(inputarg[i].c_str());
        }
        else if (inputarg[i]=="--edge") 
        {
            i++;
            arguments.edge_num=atol(inputarg[i].c_str());
        }
        else
        {
            cerr<<"wrong argument: "<<inputarg[i]<<endl;
            i++;
            //return;
        }
    }
    return;
}

//==============================================================//

void randomgraph_construction(graph_t &g, size_t vertex_num, size_t edge_num, gBenchPerf_event & perf, int perf_group)
{
    perf.open(perf_group);
    perf.start(perf_group);

    for (size_t i=0;i<vertex_num;i++) 
    {
        vertex_iterator vit = g.add_vertex();
        vit->set_property(vertex_property(i));
    }
    for (size_t i=0;i<edge_num;i++) 
    {
        edge_iterator eit;
        if (g.add_edge(rand()%vertex_num, rand()%vertex_num, eit))
            eit->set_property(edge_property(i));
    }
    perf.stop(perf_group);
}



//==============================================================//

void output(graph_t& g)
{
    cout<<"BFS Results: \n";
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        cout<<"== vertex "<<vit->id()<<": edge#-"<<vit->edges_size()<<"\n";
    }
}

//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: graph construction\n";

    arg_t arguments;
    vector<string> inputarg;
    argument_parser::initialize(argc,argv,inputarg);
    gBenchPerf_event perf(inputarg,false);
    arg_init(arguments);
    arg_parser(arguments,inputarg);

    double t1, t2;
    
    cout<<"== "<<arguments.vertex_num<<" vertices  "<<arguments.edge_num<<" edges\n";

    unsigned run_num = ceil(perf.get_event_cnt() / (double)DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;
    double elapse_time = 0;
    
    for (unsigned i=0;i<run_num;i++)
    {
        srand(SEED); // fix seed to avoid runtime dynamics

        t1 = timer::get_usec();
        graph_t g;
        randomgraph_construction(g, arguments.vertex_num, arguments.edge_num, perf, i);

        t2 = timer::get_usec();
        elapse_time += t2-t1;
    }
    cout<<"\nconstruction finish \n";

#ifndef ENABLE_VERIFY
    cout<<"== time: "<<elapse_time/run_num<<" sec\n";
    perf.print();
#endif

#ifdef ENABLE_OUTPUT
    cout<<"\n";
    output(g);
#endif

    cout<<"==================================================================\n";
    return 0;
}  // end main

