//====== Graph Benchmark Suites ======//
//======== kCore Decomposition =======//
//
// Usage: ./kcore.exe --dataset <dataset path> --kcore <k value>

#include "../lib/common.h"
#include "../lib/def.h"
#include "openG.h"
#include <queue>

using namespace std;

class vertex_property
{
public:
    vertex_property():degree(0),removed(false){}

    uint64_t degree;
    bool removed;
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
    size_t k;
};

void arg_init(arg_t& arguments)
{
    arguments.k = 3;
    arguments.dataset_path.clear();
}

void arg_parser(arg_t& arguments, vector<string>& inputarg)
{
    for (size_t i=1;i<inputarg.size();i++) 
    {

        if (inputarg[i]=="--kcore") 
        {
            i++;
            arguments.k=atol(inputarg[i].c_str());
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
size_t kcore(graph_t& g, size_t k) 
{
    size_t remove_cnt=0;

    queue<vertex_iterator> process_q;

    // initialize
    for (vertex_iterator vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        size_t degree = vit->edges_size();
        vit->property().degree = degree;

        if (degree < k) 
        {
            process_q.push(vit);
            vit->property().removed = true;
            remove_cnt++;
        }
    }

    // remove vertices iteratively 
    while (!process_q.empty()) 
    {

        vertex_iterator vit = process_q.front();
        process_q.pop();

        for (edge_iterator eit=vit->edges_begin(); eit!=vit->edges_end(); eit++) 
        {
            size_t targ = eit->target();
            vertex_iterator targ_vit = g.find_vertex(targ);
            if (targ_vit->property().removed==false)
            {
                targ_vit->property().degree--;
                if (targ_vit->property().degree < k) 
                {
                    targ_vit->property().removed = true;
                    process_q.push(targ_vit);
                    remove_cnt++;
                }
            }
        }

    }

    return remove_cnt;
}  // end kcore
//==============================================================//
void output(graph_t& g)
{
    cout<<"kCore Results: \n";
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        cout<<"== vertex "<<vit->id()<<": degree-"<<vit->property().degree
            <<" removed-";
        if (vit->property().removed) 
            cout<<"true\n";
        else
            cout<<"false\n";
    }
}

//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: kCore decomposition\n";
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

    cout<<"computing kCore: k="<<arguments.k<<"\n";
    size_t remove_cnt;
    t1 = timer::get_usec();
    perf.start();

    remove_cnt=kcore(graph, arguments.k);

    perf.stop();
    t2 = timer::get_usec();
    cout<<"== removed vertices: "<<remove_cnt<<endl;
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

