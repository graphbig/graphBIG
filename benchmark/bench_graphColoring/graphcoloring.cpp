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

#define SEED 123
#define MY_INFINITY 0xfff0
size_t beginiter = 0;
size_t enditer = 0;

class vertex_property
{
public:
    vertex_property():rand(0),color(MY_INFINITY){}

    uint16_t rand;
    uint16_t color;
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
}
//==============================================================//
inline unsigned vertex_distributor(uint64_t vid, unsigned threadnum)
{
    return vid%threadnum;
}

void init_graphcoloring(graph_t& g, unsigned threadnum,
        vector<vector<uint64_t> >& global_input_tasks)
{
    srand(SEED);
    global_input_tasks.resize(threadnum);
    for (vertex_iterator vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        vit->property().rand = rand();
        global_input_tasks[vertex_distributor(vit->id(), threadnum)].push_back(vit->id());
    }
}
void parallel_graphcoloring(graph_t& g, unsigned threadnum,
        vector<vector<uint64_t> >& global_input_tasks,
        gBenchPerf_multi & perf, int perf_group)
{
    vector<vector<uint64_t> > global_output_tasks(threadnum*threadnum);
    bool stop = false;
    #pragma omp parallel num_threads(threadnum) shared(stop,global_input_tasks,global_output_tasks) 
    {
        unsigned tid = omp_get_thread_num();
        vector<uint64_t> & input_tasks = global_input_tasks[tid];
        uint16_t color = 0; 
        perf.open(tid, perf_group);
        perf.start(tid, perf_group);  
#ifdef SIM
        unsigned iter = 0;
#endif    
        while(!stop)
        {
            #pragma omp barrier
            // process local queue
            stop = true;
            
#ifdef SIM
            SIM_BEGIN(iter==beginiter);
            iter++;
#endif       
            for (unsigned i=0;i<input_tasks.size();i++)
            {
                uint64_t vid=input_tasks[i];
                vertex_iterator vit = g.find_vertex(vid);
                uint16_t local_rand = vit->property().rand;
                bool found_larger = false;

                for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++)
                {
                    uint64_t dest_vid = eit->target();
                    vertex_iterator destvit = g.find_vertex(dest_vid);

#ifdef HMC
                    if (HMC_COMP_less(&(destvit->property().color),color)) 
                        continue;
                    if (HMC_COMP_greater(&(destvit->property().rand),local_rand))
                    {
                        found_larger = true;
                        break;
                    }
                    else if (HMC_COMP_equal(&(destvit->property().rand),local_rand) 
                            && dest_vid < vid)
                    {
                        found_larger = true;
                        break;
                    }
#else                    
                    if (destvit->property().color < color) continue;
                    if (destvit->property().rand > local_rand ||
                        (destvit->property().rand == local_rand && dest_vid < vid))
                    {
                        found_larger = true;
                        break;
                    }
#endif
                }
                if (found_larger == false)
                    vit->property().color = color;
                else
                    global_output_tasks[vertex_distributor(vid,threadnum)+tid*threadnum].push_back(vid);
            }
#ifdef SIM
            SIM_END(iter==enditer);
#endif
            #pragma omp barrier
            input_tasks.clear();
            for (unsigned i=0;i<threadnum;i++)
            {
                if (global_output_tasks[i*threadnum+tid].size()!=0)
                {
                    stop = false;
                    input_tasks.insert(input_tasks.end(),
                            global_output_tasks[i*threadnum+tid].begin(),
                            global_output_tasks[i*threadnum+tid].end());
                    global_output_tasks[i*threadnum+tid].clear();
                }
            }
            color++;
            #pragma omp barrier

        }
#ifdef SIM
        SIM_END(enditer==0);
#endif
        perf.stop(tid, perf_group);
    }

}
//==============================================================//
void output(graph_t& g)
{
    cout<<"Graph Coloring Results: \n";
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        cout<<"== vertex "<<vit->id()<<": color "<<vit->property().color<<endl;
    }
}
void reset_graph(graph_t & g)
{
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        vit->property().color = MY_INFINITY;
        vit->property().rand = 0;
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
#ifdef SIM
    arg.get_value("beginiter",beginiter);
    arg.get_value("enditer",enditer);
#endif   

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

    cout<<"computing graph color...\n";
    
    gBenchPerf_multi perf_multi(threadnum, perf);
    unsigned run_num = ceil(perf.get_event_cnt() /(double) DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;
    double elapse_time = 0;
    
    for (unsigned i=0;i<run_num;i++)
    {
        vector<vector<uint64_t> > global_input_tasks(threadnum);

        init_graphcoloring(graph,threadnum,global_input_tasks);

        t1 = timer::get_usec();

        parallel_graphcoloring(graph,threadnum,global_input_tasks,perf_multi,i);
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

