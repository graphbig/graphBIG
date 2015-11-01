//====== Graph Benchmark Suites ======//
//====== Betweenness Centrality ======//
//
// BC for unweighted graph
// Brandes' algorithm 
// Usage: ./bc.exe --dataset <dataset path>

#include "common.h"
#include "def.h"
#include "openG.h"
#include "omp.h"
#include <stack>
#include <queue>
#include <vector>
#include <list>

#ifdef HMC
#include "HMC.h"
#endif

#ifdef SIM
#include "SIM.h"
#endif

#define MY_INFINITY 0xfff0

using namespace std;

size_t maxiter = 0;

class vertex_property
{
public:
    vertex_property():BC(0){}

    double BC;
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
    arg.add_arg("undirected","1","graph directness", false);
    arg.add_arg("maxiter","0","maximum loop iteration (0-unlimited, only set for simulation purpose)");
}
//==============================================================//

void bc(graph_t& g, bool undirected,
        gBenchPerf_event & perf, int perf_group)
{
    typedef list<size_t> vertex_list_t;
    // initialization
    size_t vnum = g.num_vertices();

    vector<vertex_list_t> shortest_path_parents(vnum);
    vector<size_t> num_of_paths(vnum);
    vector<int8_t> depth_of_vertices(vnum); // 8 bits signed
    vector<double> centrality_update(vnum);

    double normalizer;
    normalizer = (undirected)? 2.0 : 1.0;

    perf.open(perf_group);
    perf.start(perf_group);


    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        size_t vertex_s = vit->id();
        stack<size_t> order_seen_stack;
        queue<size_t> BFS_queue;

        BFS_queue.push(vertex_s);

        for (size_t i=0;i<vnum;i++) 
        {
            shortest_path_parents[i].clear();

            num_of_paths[i] = (i==vertex_s) ? 1 : 0;
            depth_of_vertices[i] = (i==vertex_s) ? 0: -1;
            centrality_update[i] = 0;
        }

        // BFS traversal
        while (!BFS_queue.empty()) 
        {
            size_t v = BFS_queue.front();
            BFS_queue.pop();
            order_seen_stack.push(v);

            vertex_iterator vit = g.find_vertex(v);

            for (edge_iterator eit=vit->edges_begin(); eit!= vit->edges_end(); eit++) 
            {
                size_t w = eit->target();
                
                if (depth_of_vertices[w]<0) 
                {
                    BFS_queue.push(w);
                    depth_of_vertices[w] = depth_of_vertices[v] + 1;
                }

                if (depth_of_vertices[w] == (depth_of_vertices[v] + 1)) 
                {
                    num_of_paths[w] += num_of_paths[v];
                    shortest_path_parents[w].push_back(v);
                }
            }

        }

        // dependency accumulation
        while (!order_seen_stack.empty()) 
        {
            size_t w = order_seen_stack.top();
            order_seen_stack.pop();

            double coeff = (1+centrality_update[w])/(double)num_of_paths[w];
            vertex_list_t::iterator iter;
            for (iter=shortest_path_parents[w].begin(); 
                  iter!=shortest_path_parents[w].end(); iter++) 
            {
                size_t v=*iter;

                centrality_update[v] += (num_of_paths[v]*coeff);
            }

            if (w!=vertex_s) 
            {
                vertex_iterator vit = g.find_vertex(w);
                vit->property().BC += centrality_update[w]/normalizer;
            }
        }
    }

    perf.stop(perf_group);

    return;
}

void parallel_bc(graph_t& g, unsigned threadnum, bool undirected,
        gBenchPerf_multi & perf, int perf_group)
{
    typedef list<size_t> vertex_list_t;
    size_t vnum = g.num_vertices();
    
    uint64_t chunk = (unsigned)ceil(vnum/(double)threadnum);
    double normalizer;
    normalizer = (undirected)? 2.0 : 1.0;
    #pragma omp parallel num_threads(threadnum)
    {
        unsigned tid = omp_get_thread_num();

        perf.open(tid, perf_group);
        perf.start(tid, perf_group);  

        unsigned start = tid*chunk;
        unsigned end = start + chunk;
        if (maxiter != 0 && chunk > maxiter) end = start + maxiter;
        if (end > vnum) end = vnum;

        // initialization
        vector<vertex_list_t> shortest_path_parents(vnum);
        vector<int16_t> num_of_paths(vnum);
        vector<uint16_t> depth_of_vertices(vnum); // 16 bits signed
        vector<float> centrality_update(vnum);
#ifdef SIM
        SIM_BEGIN(true);
#endif
        for (uint64_t vid=start;vid<end;vid++) 
        {
            size_t vertex_s = vid;
            stack<size_t> order_seen_stack;
            queue<size_t> BFS_queue;

            BFS_queue.push(vertex_s);

            for (size_t i=0;i<vnum;i++) 
            {
                shortest_path_parents[i].clear();

                num_of_paths[i] = (i==vertex_s) ? 1 : 0;
                depth_of_vertices[i] = (i==vertex_s) ? 0: MY_INFINITY;
                centrality_update[i] = 0;
            }

            // BFS traversal
            while (!BFS_queue.empty()) 
            {
                size_t v = BFS_queue.front();
                BFS_queue.pop();
                order_seen_stack.push(v);

                vertex_iterator vit = g.find_vertex(v);
                uint16_t newdepth = depth_of_vertices[v]+1;
                for (edge_iterator eit=vit->edges_begin(); eit!= vit->edges_end(); eit++) 
                {
                    size_t w = eit->target();
#ifdef HMC
                    if (HMC_CAS_equal_16B(&(depth_of_vertices[w]),MY_INFINITY,newdepth) == MY_INFINITY)
                    {
                        BFS_queue.push(w);
                    }
                    if (depth_of_vertices[w] == newdepth)
                    {
                        HMC_ADD_16B(&(num_of_paths[w]), num_of_paths[v]);
                        shortest_path_parents[w].push_back(v);
                    }
#else                    
                    if (depth_of_vertices[w] == MY_INFINITY) 
                    {
                        BFS_queue.push(w);
                        depth_of_vertices[w] = newdepth;
                    }

                    if (depth_of_vertices[w] == newdepth) 
                    {
                        num_of_paths[w] += num_of_paths[v];
                        shortest_path_parents[w].push_back(v);
                    }
#endif
                }

            }

            // dependency accumulation
            while (!order_seen_stack.empty()) 
            {
                size_t w = order_seen_stack.top();
                order_seen_stack.pop();

                float coeff = (1+centrality_update[w])/(double)num_of_paths[w];
                vertex_list_t::iterator iter;
                for (iter=shortest_path_parents[w].begin(); 
                        iter!=shortest_path_parents[w].end(); iter++) 
                {
                    size_t v=*iter;
#ifdef HMC
                    HMC_FP_ADD(&(centrality_update[v]), (num_of_paths[v]*coeff));
#else
                    centrality_update[v] += (num_of_paths[v]*coeff);
#endif
                }

                if (w!=vertex_s) 
                {
                    vertex_iterator vit = g.find_vertex(w);
                    #pragma omp atomic
                    vit->property().BC += centrality_update[w]/normalizer;
                }
            }
        }
#ifdef SIM
        SIM_END(true);
#endif
        perf.stop(tid, perf_group);

    }
    return;
}
//==============================================================//
void output(graph_t& g)
{
    cout<<"Betweenness Centrality Results: \n";
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        cout<<"== vertex "<<vit->id()<<": "<<vit->property().BC<<"\n";
    }
}
void reset_graph(graph_t & g)
{
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        vit->property().BC = 0;
    }

}

//==============================================================//

int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: Betweenness Centrality\n";
    
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

    size_t threadnum;
    arg.get_value("threadnum",threadnum);
    arg.get_value("maxiter",maxiter);
    bool undirected;
    arg.get_value("undirected",undirected);


    graph_t graph;
    double t1, t2;

    //loading data
    cout<<"loading data... \n";
    if (undirected)
        cout<<"undirected graph\n";
    else
        cout<<"directed graph\n";
    
    t1 = timer::get_usec();
    string vfile = path + "/vertex.csv";
    string efile = path + "/edge.csv";

    if (graph.load_csv_vertices(vfile, true, separator, 0) == -1)
        return -1;
    if (graph.load_csv_edges(efile, true, separator, 0, 1) == -1) 
        return -1;

    size_t vertex_num = graph.num_vertices();
    size_t edge_num = graph.num_edges();
    t2 = timer::get_usec();
    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";

#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

    if (maxiter != 0 && threadnum != 1) 
        cout<<"\nenable maxiter: "<<maxiter<<" per thread";
    //processing
    cout<<"\ncomputing BC for all vertices...\n";
 
    gBenchPerf_multi perf_multi(threadnum, perf);
    unsigned run_num = ceil(perf.get_event_cnt() / (double)DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;
    double elapse_time = 0;
   
    for (unsigned i=0;i<run_num;i++)
    {
        t1 = timer::get_usec();

        if (threadnum==1)
            bc(graph,undirected,perf,i);
        else
            parallel_bc(graph,threadnum,undirected,perf_multi,i);

        t2 = timer::get_usec();
        elapse_time += t2-t1;
        if ((i+1)<run_num) reset_graph(graph);
    }
    cout<<"== finish\n";

#ifndef ENABLE_VERIFY
    cout<<"== time: "<<elapse_time/run_num<<" sec\n";
    if (threadnum == 1)
        perf.print();
    else
        perf_multi.print();
#endif

    //print output
#ifdef ENABLE_OUTPUT
    cout<<"\n";
    output(graph);
#endif
    cout<<"=================================================================="<<endl;
    return 0;
}  // end main

