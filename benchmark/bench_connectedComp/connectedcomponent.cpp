//====== Graph Benchmark Suites ======//
//======== Connected Component =======//
//
// Usage: ./connectedcomponent.exe --dataset <dataset path>

#include "../lib/common.h"
#include "../lib/def.h"
#include "openG.h"

#include <queue>

using namespace std;

class vertex_property
{
public:
    vertex_property():color(COLOR_WHITE),label(0){}

    uint8_t color;
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
void bfs_component(graph_t& g, size_t root) 
{
    std::queue<vertex_iterator> vertex_queue;

    vertex_iterator iter = g.find_vertex(root);
    if (iter == g.vertices_end()) 
        return;

    iter->property().color = COLOR_GREY;
    iter->property().label = root;

    vertex_queue.push(iter);
    while (!vertex_queue.empty()) 
    {
        vertex_iterator u = vertex_queue.front(); 
        vertex_queue.pop();  

        for (edge_iterator ei = u->edges_begin(); ei != u->edges_end(); ++ei) 
        {
            vertex_iterator v = g.find_vertex(ei->target()); 

            uint64_t v_color = v->property().color;
            if (v_color == COLOR_WHITE) 
            {
                v->property().color = COLOR_GREY;
                v->property().label = root;
                vertex_queue.push(v);
            } 
        }  // end for
        u->property().color = COLOR_BLACK;         
    }  // end while
}  // end bfs_component

size_t connected_component(graph_t& g)
{
    size_t ret=0;

    for (vertex_iterator vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        if (vit->property().color == COLOR_WHITE) 
        {
            bfs_component(g,vit->id());
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

    component_num = connected_component(graph);

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

