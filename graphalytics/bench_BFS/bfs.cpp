//====== Graph Benchmark Suites ======//
//======= Breadth-first Search =======//
//
// Usage: ./bfs.exe --dataset <dataset path> --root <root vertex id>

#include "common.h"
#include "def.h"
#include "perf.h"

#include "openG.h"
#include <queue>
#include "omp.h"

#ifdef SIM
#include "SIM.h"
#endif

#ifdef HMC
#include "HMC.h"
#endif

using namespace std;

#define MY_INFINITY 0xfff0
size_t beginiter = 0;
size_t enditer = 0;

class vertex_property
{
public:
    vertex_property():order(0),level(MY_INFINITY){}
    vertex_property(uint8_t x):order(0),level(MY_INFINITY){}

    uint64_t order;
    uint16_t level;
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
    arg.add_arg("root","0","root/starting vertex");
}
//==============================================================//

inline unsigned vertex_distributor(uint64_t vid, unsigned threadnum)
{
    return vid%threadnum;
}
void parallel_bfs(graph_t& g, size_t root, unsigned threadnum, gBenchPerf_multi & perf, int perf_group)
{
    // initializzation
    g.csr_vertex_property(root).level = 0;

    vector<vector<uint64_t> > global_input_tasks(threadnum);
    global_input_tasks[vertex_distributor(root, threadnum)].push_back(root);
    
    vector<vector<uint64_t> > global_output_tasks(threadnum*threadnum);

    bool stop = false;
    #pragma omp parallel num_threads(threadnum) shared(stop,global_input_tasks,global_output_tasks,perf) 
    {
        unsigned tid = omp_get_thread_num();
        vector<uint64_t> & input_tasks = global_input_tasks[tid];
      
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
                uint16_t curr_level = g.csr_vertex_property(vid).level;
                uint64_t edges_begin = g.csr_out_edges_begin(vid);
                uint64_t size = g.csr_out_edges_size(vid);

                for (unsigned i=0;i<size;i++)
                {
                    uint64_t dest_vid = g.csr_out_edge(edges_begin, i);
                    if (__sync_bool_compare_and_swap(&(g.csr_vertex_property(dest_vid).level), 
                                MY_INFINITY,curr_level+1))
                    {
                        global_output_tasks[vertex_distributor(dest_vid,threadnum)+tid*threadnum].push_back(dest_vid);
                    }
                }
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
    cout<<"BFS Results: \n";
    for (uint64_t vid=0;vid<g.vertex_num();vid++)
    {
        cout<<"== vertex "<<vid<<": level "<<g.csr_vertex_property(vid).level<<"\n";
    }
}

void reset_graph(graph_t & g)
{
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        vit->property().order = 0;
        vit->property().level = MY_INFINITY;
    }

}

//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: BFS\n";

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

    size_t root,threadnum;
    arg.get_value("root",root);
    arg.get_value("threadnum",threadnum);
#ifdef SIM
    arg.get_value("beginiter",beginiter);
    arg.get_value("enditer",enditer);
#endif


    graph_t graph;
    double t1, t2;

    cout<<"loading data... \n";    
    t1 = timer::get_usec();

    if (!graph.load_CSR_Graph(path))
    {
        cout<<"data loading error"<<endl;
        return 1;
    }

    size_t vertex_num = graph.vertex_num();
    size_t edge_num = graph.edge_num();
    t2 = timer::get_usec();
    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";

#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif


    cout<<"\nBFS root: "<<root<<"\n";
    
    gBenchPerf_multi perf_multi(threadnum, perf);
    unsigned run_num = ceil(perf.get_event_cnt() /(double) DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;
    double elapse_time = 0;
    
    for (unsigned i=0;i<run_num;i++)
    {
        t1 = timer::get_usec();

        parallel_bfs(graph, root, threadnum, perf_multi, i);

        t2 = timer::get_usec();
        elapse_time += t2-t1;
        if ((i+1)<run_num) reset_graph(graph);
    }
    cout<<"BFS finish: \n";

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

    cout<<"=================================================================="<<endl;
    return 0;
}  // end main

