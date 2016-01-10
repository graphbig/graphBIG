//====== Graph Benchmark Suites ======//
//========== Shortest Path ===========//
// 
// Single-source shortest path
// 
// Usage: ./sssp    --dataset <dataset path> 
//                  --root <root vertex id> 
//                  --target <target vertex id>

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

#define MY_INFINITY 0xfff0

using namespace std;
size_t beginiter = 0;
size_t enditer = 0;

class vertex_property
{
public:
    vertex_property():distance(MY_INFINITY),predecessor(MY_INFINITY),update(MY_INFINITY){}

    uint16_t distance;
    uint64_t predecessor;
    uint16_t update;
};
class edge_property
{
public:
    edge_property():weight(1){}

    uint16_t weight;
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
typedef pair<size_t,size_t> data_pair;
class comp
{
public:
    bool operator()(data_pair a, data_pair b)
    {
        return a.second > b.second;
    }
};


void sssp(graph_t& g, size_t src, gBenchPerf_event & perf, int perf_group)
{
    priority_queue<data_pair, vector<data_pair>, comp> PQ;
    
    perf.open(perf_group);
    perf.start(perf_group);
#ifdef SIM
    SIM_BEGIN(true);
#endif
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
#ifdef SIM
    SIM_END(true);
#endif
    perf.stop(perf_group);
    return;
}

inline unsigned vertex_distributor(uint64_t vid, unsigned threadnum)
{
    return vid%threadnum;
}
void parallel_sssp(graph_t& g, size_t root, unsigned threadnum, gBenchPerf_multi & perf, int perf_group)
{
    vertex_iterator rootvit=g.find_vertex(root);
    rootvit->property().distance = 0;
    rootvit->property().update = 0; 
    
    //vector<uint16_t> update(g.num_vertices(), MY_INFINITY);
    //update[root] = 0;

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
                
                uint16_t curr_dist = vit->property().distance;
                for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++)
                {
                    uint64_t dest_vid = eit->target();
                    vertex_iterator dvit = g.find_vertex(dest_vid);
                    uint16_t new_dist = curr_dist + eit->property().weight;
#ifdef HMC
                    if (HMC_CAS_greater_16B(&(dvit->property().update),new_dist) > new_dist) 
                    {
                        global_output_tasks[vertex_distributor(dest_vid,threadnum)+tid*threadnum].push_back(dest_vid);
                    }
#else
                    bool active=false;

                    // spinning lock for critical section
                    //  can be replaced as an atomicMin operation
                    while(__sync_lock_test_and_set(&(locks[dest_vid]),1));
                    if (dvit->property().update>new_dist) 
                    {
                        active = true;
                        dvit->property().update = new_dist;
                        //dvit->property().predecessor = vid;
                    }
                    __sync_lock_release(&(locks[dest_vid]));
                    
                    if (active)
                    {
                        global_output_tasks[vertex_distributor(dest_vid,threadnum)+tid*threadnum].push_back(dest_vid);
                    }
#endif
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
            for (unsigned i=0;i<input_tasks.size();i++) 
            {
                vertex_iterator vit = g.find_vertex(input_tasks[i]);
                vit->property().distance = vit->property().update;
            }
            #pragma omp barrier
        }
#ifdef SIM
        SIM_END(enditer==0);
#endif    
        perf.stop(tid, perf_group);
    }


    delete[] locks;
}

//==============================================================//
void output(graph_t& g)
{
    cout<<"Results: \n";
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
    cout<<"Benchmark: sssp shortest path\n";
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

    size_t root,threadnum;
    arg.get_value("root",root);
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
    if (graph.load_csv_edges(path, true, separator, 0, 1) == -1)
        return -1;
#endif

    size_t vertex_num = graph.num_vertices();
    size_t edge_num = graph.num_edges();
    t2 = timer::get_usec();
    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n\n";
#endif


    // sanity check
    if (graph.find_vertex(root)==graph.vertices_end()) 
    {
        cerr<<"wrong source vertex: "<<root<<endl;
        return 0;
    }

 
    cout<<"Shortest Path: source-"<<root;
    cout<<"...\n";

    gBenchPerf_multi perf_multi(threadnum, perf);
    unsigned run_num = ceil(perf.get_event_cnt() /(double) DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;
    double elapse_time = 0;
    
    for (unsigned i=0;i<run_num;i++)
    {
        t1 = timer::get_usec();

        if (threadnum==1)
            sssp(graph, root, perf, i);
        else
            parallel_sssp(graph, root, threadnum, perf_multi, i);
        
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

