//====== Graph Benchmark Suites ======//
//======== Connected Component =======//
//
// Usage: ./connectedcomponent.exe --dataset <dataset path>

#include "../lib/common.h"
#include "../lib/def.h"
#include "openG.h"
#include "omp.h"
#include <queue>

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

struct arg_t
{
    string dataset_path;
    unsigned threadnum;
};

void arg_init(arg_t& arguments)
{
    arguments.dataset_path.clear();
    arguments.threadnum = 1;
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
        else if (inputarg[i]=="--threadnum")
        {
            i++;
            arguments.threadnum=atol(inputarg[i].c_str());
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
inline unsigned vertex_distributor(size_t vid, unsigned threadnum)
{
    return vid%threadnum;
}
void parallel_bfs(graph_t& g, size_t root, unsigned threadnum)
{
    // initializzation
    vertex_iterator rootvit=g.find_vertex(root);
    if (rootvit==g.vertices_end()) return;

    rootvit->property().level = 0;
    rootvit->property().label = root;

    vector<vector<uint64_t> > global_input_tasks(threadnum);
    global_input_tasks[vertex_distributor(root, threadnum)].push_back(root);
    
    vector<vector<uint64_t> > global_output_tasks(threadnum*threadnum);

    bool stop = false;
    #pragma omp parallel num_threads(threadnum) shared(stop,global_input_tasks,global_output_tasks) 
    {
        unsigned tid = omp_get_thread_num();
        vector<uint64_t> & input_tasks = global_input_tasks[tid];
        
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
    }
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

size_t connected_component(graph_t& g, unsigned threadnum)
{
    size_t ret=0;

    for (vertex_iterator vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        if (vit->property().level == MY_INFINITY) 
        {
            if (threadnum==1) 
                bfs_component(g, vit->id());
            else
                parallel_bfs(g, vit->id(), threadnum);
            ret++;
        }
    }

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


int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: connected component\n";

    arg_t arguments;
    vector<string> inputarg;
    argument_parser::initialize(argc,argv,inputarg);
    gBenchPerf_event perf(inputarg);
    arg_init(arguments);
    arg_parser(arguments,inputarg);

    double t1, t2;
    graph_t graph;

    cout<<"loading data... \n";
    t1 = timer::get_usec();
    string vfile = arguments.dataset_path + "/vertex.csv";
    string efile = arguments.dataset_path + "/edge.csv";

    if (graph.load_csv_vertices(vfile, true, "|", 0) == -1)
        return -1;
    if (graph.load_csv_edges(efile, true, "|", 0, 1) == -1) 
        return -1;
    size_t vertex_num = graph.num_vertices();
    size_t edge_num = graph.num_edges();
    t2 = timer::get_usec();
    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

    cout<<"\ncomputing connected component...\n";
    size_t component_num;
    t1 = timer::get_usec();
    perf.start();

    component_num = connected_component(graph, arguments.threadnum);

    perf.stop();
    t2 = timer::get_usec();
    cout<<"== total component num: "<<component_num<<endl;
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
    perf.print();
#endif

#ifdef ENABLE_OUTPUT
    cout<<endl;
    output(graph);
#endif
    cout<<"==================================================================\n";
    return 0;
}  // end main

