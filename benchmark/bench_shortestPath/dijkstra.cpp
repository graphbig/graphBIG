//====== Graph Benchmark Suites ======//
//========== Shortest Path ===========//
// 
// Single-source shortest path
// 
// Usage: ./dijstra.exe --dataset <dataset path> 
//                      --root <root vertex id> 
//                      --target <target vertex id>

#include "../lib/common.h"
#include "../lib/def.h"
#include "openG.h"
#include <queue>
#include "omp.h"

#define MY_INFINITY 0xffffff00

using namespace std;

class vertex_property
{
public:
    vertex_property():distance(MY_INFINITY),predecessor(MY_INFINITY){}

    uint64_t distance;
    uint64_t predecessor;
};
class edge_property
{
public:
    edge_property():weight(1){}

    uint64_t weight;
};

typedef openG::extGraph<vertex_property, edge_property> graph_t;
typedef graph_t::vertex_iterator    vertex_iterator;
typedef graph_t::edge_iterator      edge_iterator;

//==============================================================//

struct arg_t
{
    string dataset_path;
    size_t root_vid;
    unsigned threadnum;
};

void arg_init(arg_t& arguments)
{
    arguments.root_vid = 0;
    arguments.dataset_path.clear();
    arguments.threadnum = 1;
}

void arg_parser(arg_t& arguments, vector<string>& inputarg)
{
    for (size_t i=1;i<inputarg.size();i++) 
    {

        if (inputarg[i]=="--root") 
        {
            i++;
            arguments.root_vid=atol(inputarg[i].c_str());
        }
        else if (inputarg[i]=="--threadnum")
        {
            i++;
            arguments.threadnum=atol(inputarg[i].c_str());
        }
        else if (inputarg[i]=="--dataset") 
        {
            i++;
            arguments.dataset_path=inputarg[i];
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
typedef pair<size_t,size_t> data_pair;
class comp
{
public:
    bool operator()(data_pair a, data_pair b)
    {
        return a.second > b.second;
    }
};


void dijkstra(graph_t& g, size_t src, gBenchPerf_event & perf, int perf_group)
{
    priority_queue<data_pair, vector<data_pair>, comp> PQ;
    
    perf.open(perf_group);
    perf.start(perf_group);

    // initialize
    vertex_iterator src_vit = g.find_vertex(src);
    src_vit->property().distance = 0;
    PQ.push(data_pair(src,0));

    // for every un-visited vertex, try relaxing the path
    while (!PQ.empty())
    {
        size_t u = PQ.top().first; 
        PQ.pop();

        vertex_iterator u_vit = g.find_vertex(u);

        for (edge_iterator eit = u_vit->edges_begin(); eit != u_vit->edges_end(); eit++)
        {
            size_t v = eit->target();
            vertex_iterator v_vit = g.find_vertex(v);

            size_t alt = u_vit->property().distance + eit->property().weight;
            if (alt < v_vit->property().distance) 
            {
                v_vit->property().distance = alt;
                v_vit->property().predecessor = u;
                PQ.push(data_pair(v,alt));
            }
        }
    }

    perf.stop(perf_group);
    return;
}

inline unsigned vertex_distributor(uint64_t vid, unsigned threadnum)
{
    return vid%threadnum;
}
void parallel_dijkstra(graph_t& g, size_t root, unsigned threadnum, gBenchPerf_multi & perf, int perf_group)
{
    vertex_iterator rootvit=g.find_vertex(root);
    rootvit->property().distance = 0;
    
    vector<uint32_t> update(g.num_vertices(), MY_INFINITY);
    update[root] = 0;

    bool * locks = new bool[g.num_vertices()];
    memset(locks, 0, sizeof(bool)*g.num_vertices()); 

    vector<vector<uint64_t> > global_input_tasks(threadnum);
    global_input_tasks[vertex_distributor(root,threadnum)].push_back(root);
    
    vector<vector<uint64_t> > global_output_tasks(threadnum*threadnum);

    
    bool stop = false;
    #pragma omp parallel num_threads(threadnum) shared(stop,global_input_tasks,global_output_tasks) 
    {
        unsigned tid = omp_get_thread_num();
        vector<uint64_t> & input_tasks = global_input_tasks[tid];
     
        perf.open(tid, perf_group);
        perf.start(tid, perf_group);  
        while(!stop)
        {
            #pragma omp barrier
            // process local queue
            stop = true;
            
            for (unsigned i=0;i<input_tasks.size();i++)
            {
                uint64_t vid=input_tasks[i];
                vertex_iterator vit = g.find_vertex(vid);

                uint64_t curr_dist = vit->property().distance;
                for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++)
                {
                    uint64_t dest_vid = eit->target();
                    vertex_iterator dvit = g.find_vertex(dest_vid);
                    uint32_t new_dist = curr_dist + eit->property().weight;
                    bool active=false;
                    
                    // spinning lock for critical section
                    //  can be replaced as an atomicMin operation
                    while(__sync_lock_test_and_set(&(locks[dest_vid]),1));
                    if (update[dest_vid]>new_dist) 
                    {
                        active = true;
                        update[dest_vid] = new_dist;
                        dvit->property().predecessor = vid;
                    }
                    __sync_lock_release(&(locks[dest_vid]));
                    
                    if (active)
                    {
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
            for (unsigned i=0;i<input_tasks.size();i++) 
            {
                vertex_iterator vit = g.find_vertex(input_tasks[i]);
                vit->property().distance = update[input_tasks[i]];
            }
            #pragma omp barrier
        }
        perf.stop(tid, perf_group);
    }


    delete[] locks;
}

//==============================================================//
void output(graph_t& g)
{
    cout<<"Dijkstra Results: \n";
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        cout<<"== vertex "<<vit->id()<<": distance-";
        if (vit->property().distance == MY_INFINITY) 
            cout<<"INF";
        else
            cout<<vit->property().distance;
        cout<<"\n";
    }
    return;
}

void reset_graph(graph_t & g)
{
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        vit->property().predecessor = MY_INFINITY;
        vit->property().distance = MY_INFINITY;
    }

}
//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: Dijkstra shortest path\n";
    double t1, t2;
    arg_t arguments;
    vector<string> inputarg;
    argument_parser::initialize(argc,argv,inputarg);
    gBenchPerf_event perf(inputarg, false);
    arg_init(arguments);
    arg_parser(arguments,inputarg);


    graph_t graph;
    cout<<"loading data... \n";

    t1 = timer::get_usec();
    string vfile = arguments.dataset_path + "/vertex.csv";
    string efile = arguments.dataset_path + "/edge.csv";

    if (graph.load_csv_vertices(vfile, true, "|,", 0) == -1)
        return -1;
    if (graph.load_csv_edges(efile, true, "|,", 0, 1) == -1) 
        return -1;

    size_t vertex_num = graph.num_vertices();
    size_t edge_num = graph.num_edges();
    t2 = timer::get_usec();
    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n\n";
#endif

    // input arguments
    size_t root = arguments.root_vid;

    // sanity check
    if (graph.find_vertex(root)==graph.vertices_end()) 
    {
        cerr<<"wrong source vertex: "<<root<<endl;
        return 0;
    }

 
    cout<<"Shortest Path: source-"<<root;
    cout<<"...\n";

    gBenchPerf_multi perf_multi(arguments.threadnum, perf);
    unsigned run_num = ceil(perf.get_event_cnt() /(double) DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;
    double elapse_time = 0;
    
    for (unsigned i=0;i<run_num;i++)
    {
        t1 = timer::get_usec();

        if (arguments.threadnum==1)
            dijkstra(graph, root, perf, i);
        else
            parallel_dijkstra(graph, root, arguments.threadnum, perf_multi, i);
        
        t2 = timer::get_usec();
        elapse_time += t2-t1;
    }
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<elapse_time/run_num<<" sec\n";
    if (arguments.threadnum == 1)
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

