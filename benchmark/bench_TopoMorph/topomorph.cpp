//====== Graph Benchmark Suites ======//
//======= Graph Moralization =======//
//
// Usage: ./moralization --dataset <dataset path>

#include "../lib/common.h"
#include "../lib/def.h"
#include "../lib/perf.h"
#include "openG.h"

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
};

void arg_init(arg_t& arguments)
{
    arguments.dataset_path.clear();
}

void arg_parser(arg_t& arguments, vector<string>& inputarg)
{
    for (size_t i=1; i<inputarg.size(); i++)
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

void moralize(graph_t & dag, graph_t & ug)
{
    // convert a DAG into an undirected graph
    for (vertex_iterator vit=dag.vertices_begin(); vit!=dag.vertices_end(); vit++)
    {
        ug.add_vertex(vit->property());
    }
    for (vertex_iterator vit=dag.vertices_begin(); vit!=dag.vertices_end(); vit++)
    {
        for (edge_iterator eit=vit->edges_begin(); eit!=vit->edges_end(); eit++)
        {
            edge_iterator neweit;
            if (ug.add_edge(vit->id(), eit->target(), neweit))
                neweit->set_property(eit->property());
        }
    }

    // for each node in the directed graph, make the parents
    // pairwisely connected in the undirected graph
    for (vertex_iterator vit=dag.vertices_begin(); vit!=dag.vertices_end(); vit++)
    {
        for (edge_iterator pit=vit->preds_begin(); pit!=vit->preds_end(); pit++)
        {
            edge_iterator pit2 = pit;
            pit2++;
            for (; pit2!=vit->preds_end(); pit2++)
            {
                uint64_t src = pit->target();     // src, targ are parents
                uint64_t targ = pit2->target();
                edge_iterator teit;
                if (!ug.find_out_edge(src, targ, teit))
                {
                    edge_iterator tmpit;
                    ug.add_edge(src, targ, tmpit);
                }
            }
        }
    }
}

//==============================================================//

void output(graph_t& ug, std::string path)
{
    cout<<"Writing back ugraph...\n";
    vertex_iterator vit;
    for (vit=ug.vertices_begin(); vit!=ug.vertices_end(); vit++)
    {
        cout<<"== "<<vit->id()<<": ";
        for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++)
            cout<<eit->target()<<" ";
        cout<<"\n";
    }
}

//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: Moralization\n";

    arg_t arguments;
    vector<string> inputarg;
    argument_parser::initialize(argc,argv,inputarg);
    gBenchPerf_event perf(inputarg);
    arg_init(arguments);
    arg_parser(arguments,inputarg);

    graph_t dag(openG::DIRECTED);
    double t1, t2;

    cout<<"loading data... \n";
    t1 = timer::get_usec();
    string vfile = arguments.dataset_path + "/vertex.csv";
    string efile = arguments.dataset_path + "/edge.csv";

    if (dag.load_csv_vertices(vfile, true, "|,", 0) == -1)
        return -1;
    // turn on dag_check for edge loading
    if (dag.load_csv_edges(efile, true, "|,", 0, 1, true) == -1) 
        return -1;


    size_t vertex_num = dag.num_vertices();
    size_t edge_num = dag.num_edges();
    t2 = timer::get_usec();
    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";

#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

    graph_t * ug=NULL;
    
     unsigned run_num = ceil(perf.get_event_cnt() /(double) DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;
    double elapse_time = 0;
    
    for (unsigned i=0;i<run_num;i++)
    {
        if (ug) delete ug;
        ug = new graph_t(openG::UNDIRECTED);

        t1 = timer::get_usec();
        perf.start(i);

        moralize(dag, *ug);

        perf.stop(i);
        t2 = timer::get_usec();
        elapse_time += t2-t1;
    }
    cout<<"\nMoralization finish: \n";

#ifndef ENABLE_VERIFY
    cout<<"== time: "<<elapse_time/run_num<<" sec\n";
    perf.print();
#endif

#ifdef ENABLE_OUTPUT
    cout<<"\n";
    output(*ug, arguments.dataset_path);
#endif
    if (ug) delete ug;
    cout<<"==================================================================\n";
    return 0;
}  // end main

