//====== Graph Benchmark Suites ======//
//====== Betweenness Centrality ======//
//
// BC for unweighted graph
// Brandes' algorithm 
// Usage: ./bc.exe --dataset <dataset path>

#include "../lib/common.h"
#include "../lib/def.h"
#include "openG.h"
#include "omp.h"
#include <stack>
#include <queue>
#include <vector>
#include <list>

using namespace std;

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

struct arg_t
{
    string dataset_path;
    bool skip;
    unsigned threadnum;
};

void arg_init(arg_t& arguments)
{
    arguments.dataset_path.clear();
    arguments.skip = false;
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
        else if (inputarg[i]=="--skip") 
        {
            arguments.skip = true;
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

void bc(graph_t& g, gBenchPerf_event & perf, int perf_group)
{
    typedef list<size_t> vertex_list_t;
    // initialization
    size_t vnum = g.num_vertices();

    vector<vertex_list_t> shortest_path_parents(vnum);
    vector<size_t> num_of_paths(vnum);
    vector<int8_t> depth_of_vertices(vnum); // 8 bits signed
    vector<double> centrality_update(vnum);


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

            vertex_list_t::iterator iter;
            for (iter=shortest_path_parents[w].begin(); 
                  iter!=shortest_path_parents[w].end(); iter++) 
            {
                size_t v=*iter;

                centrality_update[v] += (num_of_paths[v]/(double)num_of_paths[w])
                                        *(1+centrality_update[w]);
            }

            if (w!=vertex_s) 
            {
                vertex_iterator vit = g.find_vertex(w);
                vit->property().BC += centrality_update[w];
            }
        }
    }

    perf.stop(perf_group);

    return;
}

void parallel_bc(graph_t& g, unsigned threadnum, gBenchPerf_multi & perf, int perf_group)
{
    typedef list<size_t> vertex_list_t;
    size_t vnum = g.num_vertices();
    
    uint64_t chunk = (unsigned)ceil(vnum/(double)threadnum);
    #pragma omp parallel num_threads(threadnum)
    {
        unsigned tid = omp_get_thread_num();

        perf.open(tid, perf_group);
        perf.start(tid, perf_group);  

        unsigned start = tid*chunk;
        unsigned end = start + chunk;
        if (end > vnum) end = vnum;
        
        // initialization
        vector<vertex_list_t> shortest_path_parents(vnum);
        vector<size_t> num_of_paths(vnum);
        vector<int8_t> depth_of_vertices(vnum); // 8 bits signed
        vector<double> centrality_update(vnum);

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

                vertex_list_t::iterator iter;
                for (iter=shortest_path_parents[w].begin(); 
                        iter!=shortest_path_parents[w].end(); iter++) 
                {
                    size_t v=*iter;

                    centrality_update[v] += (num_of_paths[v]/(double)num_of_paths[w])
                        *(1+centrality_update[w]);
                }

                if (w!=vertex_s) 
                {
                    vertex_iterator vit = g.find_vertex(w);
                    #pragma omp atomic
                    vit->property().BC += centrality_update[w];
                }
            }
        }

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
    
    //argument parser
    arg_t arguments;
    vector<string> inputarg;
    argument_parser::initialize(argc,argv,inputarg);
    gBenchPerf_event perf(inputarg, false);
    arg_init(arguments);
    arg_parser(arguments,inputarg);

    graph_t graph;
    double t1, t2;

    //loading data
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
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

    if (arguments.skip) 
    {
        cout<<"\ncomputation skipped\n";
        return 0;
    }

    //processing
    cout<<"\ncomputing BC for all vertices...\n";
 
    gBenchPerf_multi perf_multi(arguments.threadnum, perf);
    unsigned run_num = ceil(perf.get_event_cnt() / (double)DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;
    double elapse_time = 0;
   
    for (unsigned i=0;i<run_num;i++)
    {
        t1 = timer::get_usec();

        if (arguments.threadnum==1)
            bc(graph,perf,i);
        else
            parallel_bc(graph, arguments.threadnum,perf_multi,i);

        t2 = timer::get_usec();
        elapse_time += t2-t1;
        reset_graph(graph);
    }
    cout<<"== finish\n";

#ifndef ENABLE_VERIFY
    cout<<"== time: "<<elapse_time/run_num<<" sec\n";
    if (arguments.threadnum == 1)
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

