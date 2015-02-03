//====== Graph Benchmark Suites ======//
//======== Degree Centrality =========//

#include "../lib/common.h"
#include "../lib/def.h"
#include "openG.h"

#include <stack>

using namespace std;


class vertex_property
{
public:
    vertex_property():indegree(0),outdegree(0){}

    uint64_t indegree;
    uint64_t outdegree;
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
};

void arg_init(arg_t& arguments)
{
    arguments.dataset_path.clear();
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
        else
        {
            cerr<<"wrong argument: "<<inputarg[i]<<endl;
            return;
        }
    }
    return;
}

//==============================================================//
void dc(graph_t& g) 
{
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
}// end dc
void degree_analyze(graph_t& g, 
                    uint64_t& indegree_max, uint64_t& indegree_min,
                    uint64_t& outdegree_max, uint64_t& outdegree_min)
{
    vertex_iterator vit;
    indegree_max=outdegree_max=0;
    indegree_min=outdegree_min=numeric_limits<uint64_t>::max();


    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        if (indegree_max < vit->property().indegree)
            indegree_max = vit->property().indegree;

        if (outdegree_max < vit->property().outdegree)
            outdegree_max = vit->property().outdegree;

        if (indegree_min > vit->property().indegree)
            indegree_min = vit->property().indegree;

        if (outdegree_min > vit->property().outdegree)
            outdegree_min = vit->property().outdegree;
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


int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: Degree Centrality\n";

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

    cout<<"\ncomputing DC for all vertices...\n";
    // Degree Centrality
    t1 = timer::get_usec();
    perf.start();
    uint64_t indegree_max, indegree_min, outdegree_max, outdegree_min;

    dc(graph);
    degree_analyze(graph, indegree_max, indegree_min, outdegree_max, outdegree_min);

    perf.stop();
    t2 = timer::get_usec();

    cout<<"DC finish: \n";
    cout<<"== inDegree[Max-"<<indegree_max<<" Min-"<<indegree_min
        <<"]  outDegree[Max-"<<outdegree_max<<" Min-"<<outdegree_min
        <<"]"<<endl;
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

