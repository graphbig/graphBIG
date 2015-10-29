//====== Graph Benchmark Suites ======//
//======== Degree Centrality =========//

#include "common.h"
#include "def.h"
#include "openG.h"
#include <math.h>
#include <stack>
#include "omp.h"

#ifdef SIM
#include "SIM.h"
#endif
#ifdef HMC
#include "HMC.h"
#endif

using namespace std;


class vertex_property
{
public:
    vertex_property():indegree(0),outdegree(0){}

    int16_t indegree;
    int16_t outdegree;
};
class edge_property
{
public:
    edge_property():value(0){}
    edge_property(uint8_t x):value(x){}

    uint8_t value;
};

typedef openG::extGraph<vertex_property, edge_property> graph_t;
typedef graph_t::vertex_iterator    vertex_iterator;
typedef graph_t::edge_iterator      edge_iterator;

//==============================================================//

//==============================================================//
void dc(graph_t& g, gBenchPerf_event & perf, int perf_group) 
{
    perf.open(perf_group);
    perf.start(perf_group);
#ifdef SIM
    SIM_BEGIN(true);
#endif
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        // out degree
        vit->property().outdegree = vit->edges_size();

        // in degree
        edge_iterator eit;
        for (eit=vit->edges_begin(); eit!=vit->edges_end(); eit++) 
        {
            vertex_iterator targ = g.find_vertex(eit->target());
            (targ->property().indegree)++;
        }
    }
#ifdef SIM
    SIM_END(true);
#endif
    perf.stop(perf_group);
}// end dc
void parallel_dc(graph_t& g, unsigned threadnum, gBenchPerf_multi & perf, int perf_group)
{
    uint64_t chunk = (unsigned)ceil(g.num_vertices()/(double)threadnum);
    #pragma omp parallel num_threads(threadnum)
    {
        unsigned tid = omp_get_thread_num();

        perf.open(tid, perf_group);
        perf.start(tid, perf_group); 
       
        unsigned start = tid*chunk;
        unsigned end = start + chunk;
        if (end > g.num_vertices()) end = g.num_vertices();
#ifdef SIM
        SIM_BEGIN(true);
#endif 
        for (unsigned vid=start;vid<end;vid++)
        {
            vertex_iterator vit = g.find_vertex(vid);
            // out degree
            vit->property().outdegree = vit->edges_size();

            // in degree
            edge_iterator eit;
            for (eit=vit->edges_begin(); eit!=vit->edges_end(); eit++) 
            {
                vertex_iterator targ = g.find_vertex(eit->target());
#ifdef HMC
                HMC_ADD_16B(&(targ->property().indegree),1);
#else  
                __sync_fetch_and_add(&(targ->property().indegree), 1);
#endif
            }

        }
#ifdef SIM
        SIM_END(true);
#endif  
        perf.stop(tid, perf_group);
    }
}
void degree_analyze(graph_t& g, 
                    uint64_t& indegree_max, uint64_t& indegree_min,
                    uint64_t& outdegree_max, uint64_t& outdegree_min)
{
    vertex_iterator vit;
    indegree_max=outdegree_max=0;
    indegree_min=outdegree_min=numeric_limits<uint64_t>::max();


    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        if (indegree_max < (uint64_t)vit->property().indegree)
            indegree_max = (uint64_t)vit->property().indegree;

        if (outdegree_max < (uint64_t)vit->property().outdegree)
            outdegree_max = (uint64_t)vit->property().outdegree;

        if (indegree_min > (uint64_t)vit->property().indegree)
            indegree_min = (uint64_t)vit->property().indegree;

        if (outdegree_min > (uint64_t)vit->property().outdegree)
            outdegree_min = (uint64_t)vit->property().outdegree;
    }

    return;
}
void output(graph_t& g) 
{
    cout<<"Degree Centrality Results: \n";
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        cout<<"== vertex "<<vit->id()<<": in-"<<vit->property().indegree
            <<" out-"<<vit->property().outdegree<<"\n";
    }
}
void reset_graph(graph_t & g)
{
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        vit->property().indegree = 0;
        vit->property().outdegree = 0;
    }
}


int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: Degree Centrality\n";

    argument_parser arg;
    gBenchPerf_event perf;
    if (arg.parse(argc,argv,perf,false)==false)
    {
        arg.help();
        return -1;
    }
    string path, separator;
    arg.get_value("dataset",path);
    arg.get_value("separator",separator);

    size_t threadnum;
    arg.get_value("threadnum",threadnum);

    double t1, t2;
    graph_t graph;
    cout<<"loading data... \n";
    t1 = timer::get_usec();
    string vfile = path + "/vertex.csv";
    string efile = path + "/edge.csv";

#ifndef EDGES_ONLY
    if (graph.load_csv_vertices(vfile, true, separator, 0) == -1)
        return -1;
    if (graph.load_csv_edges(efile, true, separator, 0, 1) == -1) 
        return -1;
#else
    if (graph.load_csv_edges(path, true, separator, 0, 1) == -1)
        return -1;
#endif
   
    size_t vertex_num = graph.num_vertices();
    size_t edge_num = graph.num_edges();
    t2 = timer::get_usec();
    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

    cout<<"\ncomputing DC for all vertices...\n";

    gBenchPerf_multi perf_multi(threadnum, perf);
    unsigned run_num = ceil(perf.get_event_cnt() / (double)DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;
    double elapse_time = 0;
    
    for (unsigned i=0;i<run_num;i++)
    {
        // Degree Centrality
        t1 = timer::get_usec();
        
        if (threadnum==1)
            dc(graph, perf, i);
        else
            parallel_dc(graph, threadnum, perf_multi, i);

        t2 = timer::get_usec();
        elapse_time += t2-t1;
        if ((i+1)<run_num) reset_graph(graph);
    }

    uint64_t indegree_max, indegree_min, outdegree_max, outdegree_min;
    degree_analyze(graph, indegree_max, indegree_min, outdegree_max, outdegree_min);

    cout<<"DC finish: \n";
    cout<<"== inDegree[Max-"<<indegree_max<<" Min-"<<indegree_min
        <<"]  outDegree[Max-"<<outdegree_max<<" Min-"<<outdegree_min
        <<"]"<<endl;
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<elapse_time/run_num<<" sec\n";
    if (threadnum == 1)
        perf.print();
    else
        perf_multi.print();

#endif

#ifdef ENABLE_OUTPUT
    cout<<endl;
    output(graph);
#endif
    cout<<"==================================================================\n";
    return 0;
}  // end main

