//====== Graph Benchmark Suites ======//
//======= Breadth-first Search =======//
//
// Usage: ./bfs.exe --dataset <dataset path> --root <root vertex id>

#include "../lib/common.h"
#include "../lib/def.h"
#include "../lib/perf.h"
#include "openG.h"
#include <queue>

using namespace std;

class vertex_property
{
public:
    vertex_property():color(COLOR_WHITE),order(0){}
    vertex_property(uint8_t x):color(x),order(0){}

    uint8_t color;
    uint64_t order;
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
    size_t root_vid;
};

void arg_init(arg_t& arguments)
{
    arguments.root_vid = 0;
    arguments.dataset_path.clear();
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

class BFSVisitor
{
public:
    void white_vertex(vertex_iterator vit){white_access++;}
    void grey_vertex(vertex_iterator vit){grey_access++;}
    void black_vertex(vertex_iterator vit){black_access++;}
    void finish_vertex(vertex_iterator vit){}

    size_t white_access;
    size_t grey_access;
    size_t black_access;

    BFSVisitor()
    {
        white_access=0;
        grey_access=0;
        black_access=0;
    }
};


void bfs(graph_t& g, size_t root, BFSVisitor& vis) 
{
    std::queue<vertex_iterator> vertex_queue;

    vertex_iterator iter = g.find_vertex(root);
    if (iter == g.vertices_end()) 
        return;

    vis.white_vertex(iter);
    size_t visit_cnt=0;

    iter->property().color = COLOR_GREY;
    iter->property().order = 0;

    vertex_queue.push(iter);
    visit_cnt++;

    while (!vertex_queue.empty()) 
    {
        vertex_iterator u = vertex_queue.front(); 
        vertex_queue.pop();  

        for (edge_iterator ei = u->edges_begin(); ei != u->edges_end(); ++ei) 
        {
            vertex_iterator v = g.find_vertex(ei->target()); 


            uint8_t v_color = v->property().color;

            if (v_color == COLOR_WHITE) 
            {
                vis.white_vertex(v);

                v->property().color = COLOR_GREY;
                v->property().order = visit_cnt;

                vertex_queue.push(v);
                visit_cnt++;
            } 
            else if (v_color == COLOR_GREY) 
            {
                vis.grey_vertex(v);
            }
            else
            {
                vis.black_vertex(v);
            }
        }  // end for
        vis.finish_vertex(u);
        u->property().color = COLOR_BLACK;         

    }  // end while
}  // end bfs
//==============================================================//

void output(graph_t& g)
{
    cout<<"BFS Results: \n";
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        cout<<"== vertex "<<vit->id()<<": order "<<vit->property().order<<"\n";
    }
}

//==============================================================//
int main(int argc, char * argv[])
{
    //cout<<"===============Graph uBenchmark Suites===============\n";
    graphBIG::print();
    cout<<"Benchmark: BFS\n";

    arg_t arguments;
    vector<string> inputarg;
    argument_parser::initialize(argc,argv,inputarg);
    gBenchPerf_event perf(inputarg);
    arg_init(arguments);
    arg_parser(arguments,inputarg);

    graph_t graph;
    double t1, t2;

    cout<<"loading data... \n";    
    t1 = timer::get_usec();
    string vfile = arguments.dataset_path + "/vertex.csv";
    string efile = arguments.dataset_path + "/edge.csv";

    if (graph.load_csv_vertices(vfile, true, "|", 0) == -1)
        return -1;
    if (graph.load_csv_edges(efile, true, "|", 0, 1) == -1) 
        return -1;

    size_t vertex_num = graph.vertex_num();
    size_t edge_num = graph.edge_num();
    t2 = timer::get_usec();
    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";

#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

    size_t root=arguments.root_vid; 
    BFSVisitor vis;

    cout<<"\nBFS root: "<<root<<"\n";

    t1 = timer::get_usec();
    perf.start();

    bfs(graph, root, vis);

    perf.stop();
    t2 = timer::get_usec();

    cout<<"BFS finish: \n";
    cout<<"== w-"<<vis.white_access<<" g-"<<vis.grey_access<<" b-"<<vis.black_access<<endl;

#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
    perf.print();
#endif

#ifdef ENABLE_OUTPUT
    cout<<"\n";
    output(graph);
#endif

    cout<<"=================================================================="<<endl;
    return 0;
}  // end main

