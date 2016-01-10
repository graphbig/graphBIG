//====== Graph Benchmark Suites ======//
//======= RandomGraph Construction =======//
//
// Usage: ./randomgraph --vertex <vertex #> --edge <edge #>

#include "common.h"
#include "def.h"
#include "perf.h"
#include "openG.h"
#include "omp.h"
#ifdef SIM
#include "SIM.h"
#endif
using namespace std;

#define SEED 111
unsigned threadnum;

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
    arg.add_arg("vertex","100","vertex #");
    arg.add_arg("edge","1000","edge #");
}
//==============================================================//

void randomgraph_construction(graph_t &g, size_t vertex_num, size_t edge_num, gBenchPerf_event & perf, int perf_group)
{
    vector<pair<size_t,size_t> > edges;
    for (size_t i=0;i<edge_num;i++) 
    {
        edges.push_back(make_pair(rand()%vertex_num, rand()%vertex_num));
    }
    perf.open(perf_group);
    perf.start(perf_group);
    for (size_t i=0;i<vertex_num;i++) 
    {
        vertex_iterator vit = g.add_vertex();
        vit->set_property(vertex_property(i));
    }
#ifdef SIM
    SIM_BEGIN(true);
#endif 
    for (size_t i=0;i<edge_num;i++) 
    {
        edge_iterator eit;
        g.add_edge(edges[i].first, edges[i].second, eit);
#ifndef SIM
        eit->set_property(edge_property(i));
#endif
    }
#ifdef SIM
    SIM_END(true);
#endif 
    perf.stop(perf_group);
}

void parallel_randomgraph_construction(graph_t &g, size_t vertex_num, size_t edge_num)
{
    vector<pair<size_t,size_t> > edges;
    for (size_t i=0;i<edge_num;i++) 
    {
        edges.push_back(make_pair(rand()%vertex_num, rand()%vertex_num));
    }
    for (size_t i=0;i<vertex_num;i++) 
    {
        vertex_iterator vit = g.add_vertex();
        vit->set_property(vertex_property(i));
    }
    uint64_t chunk = (unsigned)ceil(edge_num/(double)threadnum);
    #pragma omp parallel num_threads(threadnum)
    {
        unsigned tid = omp_get_thread_num();
       
        unsigned start = tid*chunk;
        unsigned end = start + chunk;
        if (end > edge_num) end = edge_num;
#ifdef SIM
        SIM_BEGIN(true);
#endif 
        for (size_t i=start;i<end;i++) 
        {
            edge_iterator eit;
            g.add_edge(edges[i].first, edges[i].second, eit);
#ifndef SIM
            eit->set_property(edge_property(i));
#endif
        }
#ifdef SIM
        SIM_END(true);
#endif 

    }
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
    
    argument_parser arg;
    gBenchPerf_event perf;
    arg_init(arg);
    if (arg.parse(argc,argv,perf,false)==false)
    {
        arg.help();
        return -1;
    }
    
    size_t vertex_num,edge_num;
    arg.get_value("vertex",vertex_num);
    arg.get_value("edge",edge_num);
    
    arg.get_value("threadnum",threadnum);
    
    double t1, t2;
    
    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";

    unsigned run_num = ceil(perf.get_event_cnt() / (double)DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;
    double elapse_time = 0;
    
    for (unsigned i=0;i<run_num;i++)
    {
        srand(SEED); // fix seed to avoid runtime dynamics

        t1 = timer::get_usec();
        graph_t g;
        if (threadnum==1)
            randomgraph_construction(g, vertex_num, edge_num, perf, i);
        else
            parallel_randomgraph_construction(g, vertex_num, edge_num);
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

