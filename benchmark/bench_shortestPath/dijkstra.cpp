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

using namespace std;

class vertex_property
{
public:
    vertex_property():root(0),distance(INFu64),predecessor(INFu64){}

    uint64_t root;

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
    size_t targ_vid;
};

void arg_init(arg_t& arguments)
{
    arguments.root_vid = 0;
    arguments.dataset_path.clear();
    arguments.targ_vid = INFu64;
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
        else if (inputarg[i]=="--target") 
        {
            i++;
            arguments.targ_vid=atol(inputarg[i].c_str());
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


void dijkstra(graph_t& g, size_t src, size_t targ)
{
    priority_queue<data_pair, vector<data_pair>, comp> PQ;
    
    // initialize
    for (vertex_iterator vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        vit->property().root = src;
        vit->property().distance = INFu64;
        vit->property().predecessor = INFu64;
    }

    vertex_iterator src_vit = g.find_vertex(src);
    src_vit->property().root = src;
    src_vit->property().distance = 0;
    PQ.push(data_pair(src,0));

    // for every un-visited vertex, try relaxing the path
    while (!PQ.empty())
    {
        size_t u = PQ.top().first; 
        PQ.pop();
        if (targ == u) break;

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

    return;
}
//==============================================================//
void output(graph_t& g)
{
    cout<<"Dijkstra Results: \n";
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        cout<<"== vertex "<<vit->id()<<": distance-";
        if (vit->property().distance == INFu64) 
            cout<<"INF";
        else
            cout<<vit->property().distance;
        cout<<" predecessor-";
        if (vit->property().predecessor == INFu64)
            cout<<"UNDEFINED";
        else
            cout<<vit->property().predecessor;

        cout<<"\n";
    }
    return;
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
    gBenchPerf_event perf(inputarg);
    arg_init(arguments);
    arg_parser(arguments,inputarg);


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
    cout<<"== time: "<<t2-t1<<" sec\n\n";
#endif

    // input arguments
    size_t root = arguments.root_vid;
    size_t targ = arguments.targ_vid;

    // sanity check
    if (graph.find_vertex(root)==graph.vertices_end()) 
    {
        cerr<<"wrong source vertex: "<<root<<endl;
        return 0;
    }

 
    cout<<"Shortest Path: source-"<<root;
    if (targ != INFu64)
        cout<<"  target-"<<targ;
    cout<<"...\n";
    t1 = timer::get_usec();
    perf.start();

    dijkstra(graph, root, targ);

    perf.stop();
    t2 = timer::get_usec();

#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
    perf.print();
#endif


#ifdef ENABLE_OUTPUT
    cout<<"\n";
    output(graph);
#endif

    cout<<"==================================================================\n";
    return 0;
}  // end main

