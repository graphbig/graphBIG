//====== Graph Benchmark Suites ======//
//======= RandomGraph Construction =======//
//
// Usage: ./graphupdate --delete <vertex #> --dataset <dataset path>

#include "../lib/common.h"
#include "../lib/def.h"
#include "../lib/perf.h"
#include "openG.h"

using namespace std;

#define SEED 111

class vertex_property
{
public:
    vertex_property():value(0){}
    vertex_property(uint64_t x):value(x){}

    uint64_t value;
};
class edge_property
{
public:
    edge_property():value(0){}
    edge_property(uint64_t x):value(x){}

    uint64_t value;
};

typedef openG::extGraph<vertex_property, edge_property> graph_t;
typedef graph_t::vertex_iterator    vertex_iterator;
typedef graph_t::edge_iterator      edge_iterator;

//==============================================================//

struct arg_t
{
    string dataset_path;
    size_t delete_num;
};

void arg_init(arg_t& arguments)
{
    arguments.dataset_path.clear();
    arguments.delete_num = 10;
}

void arg_parser(arg_t& arguments, vector<string>& inputarg)
{
    for (size_t i=1;i<inputarg.size();i++) 
    {

        if (inputarg[i]=="--delete") 
        {
            i++;
            arguments.delete_num=atol(inputarg[i].c_str());
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

size_t input(string path, size_t num, vector<uint64_t> & q)
{
    string randfn = path + "/id.rand";
    ifstream ifs(randfn);
    if (ifs.is_open()==false)
        return 0;

    q.clear();

    size_t cnt=0;

    while (ifs.good())
    {
        string line;
        getline(ifs,line);
        
        if (line.empty()) continue;
        if (line[0]=='#') continue;

        q.push_back(atoll(line.c_str()));
        cnt++;
        if (cnt >= num) break;
    }

    ifs.close();
    return cnt;
}

//==============================================================//

void graph_update(graph_t &g, vector<uint64_t> IDs)
{
    for (size_t i=0;i<IDs.size();i++) 
    {
        if (g.num_vertices()==0) break;

        g.delete_vertex(IDs[i]);
    }
    
}

//==============================================================//

void output(graph_t& g)
{
    cout<<"Results: \n";
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        cout<<"== vertex "<<vit->id()<<": edge#-"<<vit->edges_size()<<"\n";
    }
}

//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: Graph update\n";

    arg_t arguments;
    vector<string> inputarg;
    argument_parser::initialize(argc,argv,inputarg);
    gBenchPerf_event perf(inputarg);
    arg_init(arguments);
    arg_parser(arguments,inputarg);

    srand(SEED); // fix seed to avoid runtime dynamics
    graph_t g;
    double t1, t2;
    
    cout<<"loading data... \n";

    t1 = timer::get_usec();
    string vfile = arguments.dataset_path + "/vertex.csv";
    string efile = arguments.dataset_path + "/edge.csv";

    if (g.load_csv_vertices(vfile, true, "|", 0) == -1)
        return -1;
    if (g.load_csv_edges(efile, true, "|", 0, 1) == -1) 
        return -1;

    size_t vertex_num = g.num_vertices();
    size_t edge_num = g.num_edges();
    t2 = timer::get_usec();

    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
    
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n\n";
#endif

    vector<uint64_t> IDs;
    if (input(arguments.dataset_path, arguments.delete_num, IDs)
            !=arguments.delete_num)
    {
        cout<<"Error in ID file\n";
        return -1;
    }

    t1 = timer::get_usec();
    perf.start();

    graph_update(g, IDs);


    perf.stop();
    t2 = timer::get_usec();
    cout<<"graph update finish: \n";
    cout<<"== "<<g.num_vertices()<<" vertices  "<<g.num_edges()<<" edges\n";

#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
    perf.print();
#endif

#ifdef ENABLE_OUTPUT
    cout<<"\n";
    output(g);
#endif

    cout<<"==================================================================\n";
    return 0;
}  // end main

