//====== Graph Benchmark Suites ======//
//======== Depth-first Search ========//
//
// Usage: ./dfs.exe --dataset <dataset path> --root <root vertex id>

#include "common.h"
#include "def.h"
#include "openG.h"
#include <stack>
#ifdef SIM
#include "SIM.h"
#endif

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

void arg_init(argument_parser & arg)
{
    arg.add_arg("root","0","root/starting vertex");
}

//==============================================================//

class DFSVisitor
{
public:
    void white_vertex(vertex_iterator vit){white_access++;}
    void grey_vertex(vertex_iterator vit){grey_access++;}
    void black_vertex(vertex_iterator vit){black_access++;}
    void finish_vertex(vertex_iterator vit){}

    size_t white_access;
    size_t grey_access;
    size_t black_access;

    DFSVisitor()
    {
        white_access=0;
        grey_access=0;
        black_access=0;
    }
};


void dfs(graph_t& g, size_t root, DFSVisitor& vis, gBenchPerf_event & perf, int perf_group) 
{
    perf.open(perf_group);
    perf.start(perf_group);
#ifdef SIM
    SIM_BEGIN(true);
#endif
    std::stack<vertex_iterator> vertex_stack;
    size_t visit_cnt=0;    
    
    vertex_iterator iter = g.find_vertex(root);
    if (iter == g.vertices_end()) 
        return;

    vis.white_vertex(iter);
    iter->property().color = COLOR_GREY;
    iter->property().order = 0;

    vertex_stack.push(iter);
    visit_cnt++;
    while (!vertex_stack.empty()) 
    {
        vertex_iterator u = vertex_stack.top(); 
        vertex_stack.pop();  

        for (edge_iterator ei = u->edges_begin(); ei != u->edges_end(); ++ei) 
        {
            vertex_iterator v = g.find_vertex(ei->target()); 

            uint64_t v_color = v->property().color;
            if (v_color == COLOR_WHITE) 
            {
                vis.white_vertex(v);
                v->property().color = COLOR_GREY;
                v->property().order = visit_cnt;
                vertex_stack.push(v);
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
#ifdef SIM
    SIM_END(true);
#endif
    perf.stop(perf_group);
}  // end dfs

//==============================================================//
void output(graph_t& g)
{
    cout<<"DFS Results: \n";
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        cout<<"== vertex "<<vit->id()<<": order "<<vit->property().order<<"\n";
    }
}

void reset_graph(graph_t & g)
{
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        vit->property().color = COLOR_WHITE;
        vit->property().order = 0;
    }

}

//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: DFS\n";

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

    size_t root;
    arg.get_value("root",root);

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
    if (graph.load_csv_edges(efile, true, separator, 0, 1) == -1)
        return -1;
#endif
    
    size_t vertex_num = graph.num_vertices();
    size_t edge_num = graph.num_edges();
    t2 = timer::get_usec();

    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";

#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n\n";
#endif

    DFSVisitor vis;

    cout<<"DFS root: "<<root<<"\n\n";

    unsigned run_num = ceil(perf.get_event_cnt() / (double)DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;
    double elapse_time = 0;
    
    for (unsigned i=0;i<run_num;i++)
    {
        vis.white_access=0;
        vis.grey_access=0;
        vis.black_access=0;

        t1 = timer::get_usec();

        dfs(graph, root, vis, perf, i);

        t2 = timer::get_usec();
        elapse_time += t2-t1;
        if ((i+1)<run_num) reset_graph(graph);
    }
    cout<<"DFS finish: \n";
    cout<<"== w-"<<vis.white_access<<" g-"<<vis.grey_access<<" b-"<<vis.black_access<<endl;
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<elapse_time/run_num<<" sec\n";
    perf.print();
#endif

#ifdef ENABLE_OUTPUT
    cout<<"\n";
    output(graph);
#endif
    cout<<"==================================================================\n";
    return 0;
}  // end main

