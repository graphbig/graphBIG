//====== Graph Benchmark Suites ======//
//======== kCore Decomposition =======//
//
// Usage: ./kcore.exe --dataset <dataset path> --kcore <k value>

#include "common.h"
#include "def.h"
#include "openG.h"
#include <queue>
#include "omp.h"

#ifdef HMC
#include "HMC.h"
#endif

#ifdef SIM
#include "SIM.h"
#endif

using namespace std;

class vertex_property
{
public:
    vertex_property():degree(0),core(0){}

    int16_t degree;
    uint16_t core;
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
void arg_init(argument_parser & arg)
{
    arg.add_arg("kcore","3","kCore k value");
}
//==============================================================//
inline unsigned vertex_distributor(uint64_t vid, unsigned threadnum)
{
    return vid%threadnum;
}

void seq_init(graph_t& g)
{
    // initialize
    for (vertex_iterator vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        size_t degree = vit->edges_size();
        vit->property().degree = (int16_t)degree;
    }
}
void kcore(graph_t& g, size_t k,
        gBenchPerf_event & perf, int perf_group) 
{
    perf.open(perf_group);
    perf.start(perf_group);

    // remove vertices iteratively 
    for (unsigned iter=1;iter<=k;iter++) 
    {
        for (vertex_iterator vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
        {
            if (vit->property().core != 0) continue;
            if (vit->property().degree > (int16_t)iter) continue;

            vit->property().core = iter;
            for (edge_iterator eit=vit->edges_begin(); eit!=vit->edges_end(); eit++) 
            {
                size_t targ = eit->target();
                vertex_iterator targ_vit = g.find_vertex(targ);
                targ_vit->property().degree--;
            }
        }

    }

    perf.stop(perf_group);
}  // end kcore

void parallel_kcore(graph_t& g, size_t k, unsigned threadnum,
        gBenchPerf_multi & perf, int perf_group)
{
    unsigned chunk = (unsigned) ceil(g.num_vertices()/(double)threadnum);
    #pragma omp parallel num_threads(threadnum) 
    {
        unsigned tid = omp_get_thread_num();
        unsigned start = tid*chunk;
        unsigned end = start + chunk;
        if (end > g.num_vertices()) end = g.num_vertices();

        perf.open(tid, perf_group);
        perf.start(tid, perf_group);  
#ifdef SIM
        SIM_BEGIN(true);
#endif   
        for (unsigned iter=1;iter<=k;iter++) 
        {
            #pragma omp barrier
            
            for (unsigned vid=start;vid<end;vid++)
            {
                vertex_iterator vit = g.find_vertex(vid);
                if (vit->property().core != 0) continue;
                if (vit->property().degree > (int16_t)iter) continue;

                vit->property().core = iter;

                for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++)
                {
                    uint64_t dest_vid = eit->target();
                    vertex_iterator destvit = g.find_vertex(dest_vid);
#ifdef HMC
                    HMC_ADD_16B(&(destvit->property().degree), -1); 
#else
                    __sync_fetch_and_sub(&(destvit->property().degree), 1);
#endif                        
                }
            }
        }
#ifdef SIM
        SIM_END(true);
#endif
        perf.stop(tid, perf_group);
    }

}
//==============================================================//
void output(graph_t& g)
{
    cout<<"kCore Results: \n";
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        cout<<"== vertex "<<vit->id()<<": core-"<<vit->property().core<<"\n";
    }
}
void reset_graph(graph_t & g)
{
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        vit->property().degree = 0;
        vit->property().core = 0;
    }

}

//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: kCore decomposition\n";
    double t1, t2;

    argument_parser arg;
    gBenchPerf_event perf;
    arg_init(arg);
    if (arg.parse(argc,argv,perf,false)==false)
    {
        arg.help();
        return -1;
    }
    string path, separator;
    arg.get_value("dataset",path);
    arg.get_value("separator",separator);

    size_t k,threadnum;
    arg.get_value("kcore",k);
    arg.get_value("threadnum",threadnum);
    
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
    if (graph.load_csv_edges(efile, true, separator, 0, 1) == -1)
        return -1;
#endif

    size_t vertex_num = graph.num_vertices();
    size_t edge_num = graph.num_edges();
    t2 = timer::get_usec();

    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
    
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n\n";
#endif

    cout<<"computing kCore: k="<<k<<"\n";

    gBenchPerf_multi perf_multi(threadnum, perf);
    unsigned run_num = ceil(perf.get_event_cnt() /(double) DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;
    double elapse_time = 0;
    
    for (unsigned i=0;i<run_num;i++)
    {
        seq_init(graph);

        t1 = timer::get_usec();

        if (threadnum==1)
            kcore(graph, k, perf, i);
        else
            parallel_kcore(graph, k, threadnum, perf_multi, i);
        t2 = timer::get_usec();
        elapse_time += t2-t1;
        if ((i+1)<run_num) reset_graph(graph);
    }
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<elapse_time/run_num<<" sec\n";
    if (threadnum == 1)
        perf.print();
    else
        perf_multi.print();

#endif

#ifdef ENABLE_OUTPUT
    cout<<"\n";
    output(graph);
#endif
    cout<<"==================================================================\n";
    return 0;
}  // end main

