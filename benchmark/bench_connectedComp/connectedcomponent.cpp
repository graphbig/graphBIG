//====== Graph Benchmark Suites ======//
//======== Connected Component =======//
//
// Usage: ./connectedcomponent.exe --dataset <dataset path>

#include "common.h"
#include "def.h"
#include "openG.h"
#include "omp.h"
#include <queue>
#ifdef SIM
#include "SIM.h"
#endif

#define EDGE_MARK 1
#define MY_INFINITY 0xffffff00

using namespace std;


class vertex_property
{
public:
    vertex_property():level(MY_INFINITY),label(0){}

    unsigned level;
    uint64_t label;
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
inline unsigned vertex_distributor(size_t vid, unsigned threadnum)
{
    return vid%threadnum;
}
unsigned parallel_cc(graph_t& g, unsigned threadnum, gBenchPerf_multi & perf, int perf_group)
{
    unsigned ret=0;
    // initializzation
    vector<vector<uint64_t> > global_input_tasks(threadnum);
    vector<vector<uint64_t> > global_output_tasks(threadnum*threadnum);

    uint64_t root = 0;
    bool stop = false;
    #pragma omp parallel num_threads(threadnum) shared(stop,global_input_tasks,global_output_tasks) 
    {
        unsigned tid = omp_get_thread_num();
        vector<uint64_t> & input_tasks = global_input_tasks[tid];
        
        perf.open(tid, perf_group);
        perf.start(tid, perf_group); 
        
        while(root < g.num_vertices())
        {
            if (tid == 0)
            {
                vertex_iterator rootvit=g.find_vertex(root);

                rootvit->property().level = 0;
                rootvit->property().label = root;
                global_input_tasks[vertex_distributor(root, threadnum)].push_back(root);
                stop = false;
            }
            #pragma omp barrier
            while(!stop)
            {
                #pragma omp barrier
                // process local queue
                stop = true;
                
            
                for (unsigned i=0;i<input_tasks.size();i++)
                {
                    uint64_t vid=input_tasks[i];
                    vertex_iterator vit = g.find_vertex(vid);
                    uint32_t curr_level = vit->property().level;
                    
                    for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++)
                    {
                        uint64_t dest_vid = eit->target();
                        vertex_iterator destvit = g.find_vertex(dest_vid);
                        if (__sync_bool_compare_and_swap(&(destvit->property().level), 
                                    MY_INFINITY,curr_level+1))
                        {
                            destvit->property().label = root;
                            global_output_tasks[vertex_distributor(dest_vid,threadnum)+tid*threadnum].push_back(dest_vid);
                        }
                    }
                }
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

            if (tid == 0)
            {
                ret++;

                for (;root<g.num_vertices();root++)
                {
                    vertex_iterator vit = g.find_vertex(root);
                    if (vit->property().level==MY_INFINITY) break;
                }
            }
            #pragma omp barrier
        }
        perf.stop(tid, perf_group);
    }

    return ret;
}
void bfs_component(graph_t& g, size_t root) 
{
    std::queue<vertex_iterator> vertex_queue;

    vertex_iterator iter = g.find_vertex(root);
    if (iter == g.vertices_end()) 
        return;

    iter->property().level = 0;
    iter->property().label = root;

    vertex_queue.push(iter);
    while (!vertex_queue.empty()) 
    {
        vertex_iterator u = vertex_queue.front(); 
        vertex_queue.pop();  

        for (edge_iterator ei = u->edges_begin(); ei != u->edges_end(); ++ei) 
        {
            vertex_iterator v = g.find_vertex(ei->target()); 

            if (v->property().level == MY_INFINITY) 
            {
                v->property().level = u->property().level+1;
                v->property().label = root;
                vertex_queue.push(v);
            } 
        }  // end for
    }  // end while
}  // end bfs_component

size_t connected_component(graph_t& g, gBenchPerf_event & perf, int perf_group)
{
    size_t ret=0;

    perf.open(perf_group);
    perf.start(perf_group);
#ifdef SIM
    SIM_BEGIN(true);
#endif
    for (vertex_iterator vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        if (vit->property().level == MY_INFINITY) 
        {
            bfs_component(g, vit->id());
            ret++;
        }
    }
#ifdef SIM
    SIM_END(true);
#endif
    perf.stop(perf_group);

    return ret;
}
void output(graph_t& g)
{
    cout<<"Connected Component Results: \n";
    for (vertex_iterator vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        cout<<"== vertex "<<vit->id()<<": component "<<vit->property().label<<endl;
    }
}

void reset_graph(graph_t & g)
{
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        vit->property().label = 0;
        vit->property().level = MY_INFINITY;
    }

}


int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: connected component\n";

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

    cout<<"\ncomputing connected component...\n";
    size_t component_num;
    
    gBenchPerf_multi perf_multi(threadnum, perf);
    unsigned run_num = ceil(perf.get_event_cnt() / (double)DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;
    double elapse_time = 0;

    for (unsigned i=0;i<run_num;i++)
    {
        t1 = timer::get_usec();

        if (threadnum == 1)
            component_num = connected_component(graph, perf, i);
        else
            component_num = parallel_cc(graph, threadnum, perf_multi, i);

        t2 = timer::get_usec();
        elapse_time += t2-t1;
        if ((i+1)<run_num) reset_graph(graph);
    }
    cout<<"== total component num: "<<component_num<<endl;
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

