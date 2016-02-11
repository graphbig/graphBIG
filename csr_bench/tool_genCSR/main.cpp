//====== Graph Benchmark Suites ======//
//

#include <vector>
#include <string>
#include <fstream>
#include "common.h"
#include "def.h"
#include "openG.h"

using namespace std;

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
void arg_init(argument_parser & arg)
{
    arg.add_arg("outpath","./","path for generated graph");
}
//==============================================================//


//==============================================================//


//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Tool: CSR data generation\n";

    argument_parser arg;
    arg_init(arg);
#ifndef NO_PERF    
    gBenchPerf_event perf;
    if (arg.parse(argc,argv,perf,false)==false)
    {
        arg.help();
        return -1;
    }
#else
    if (arg.parse(argc,argv,NULL,false)==false)
    {
        arg.help();
        return -1;
    }
#endif

    string path, separator, outpath;
    arg.get_value("dataset",path);
    arg.get_value("separator",separator);
    arg.get_value("outpath",outpath);

    graph_t g;
    double t1, t2;
    
    cout<<"loading data... \n";

    t1 = timer::get_usec();
    string vfile = path + "/vertex.csv";
    string efile = path + "/edge.csv";

#ifndef EDGES_ONLY
    if (g.load_csv_vertices(vfile, true, separator, 0) == -1)
        return -1;
#endif
    if (g.load_csv_edges(efile, true, separator, 0, 1) == -1) 
        return -1;

    size_t vertex_num = g.num_vertices();
    size_t edge_num = g.num_edges();
    t2 = timer::get_usec();

    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
    
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

    t1 = timer::get_usec();
    //================================================//
    vector<uint64_t> vertexlist, edgelist; 
    g.to_CSR_Graph(vertexlist, edgelist);
    t2 = timer::get_usec();

    cout<<"== data conversion time: "<<t2-t1<<" sec\n"<<endl;
    //================================================//
    

    t1 = timer::get_usec();
    //================================================//
    string vl = outpath + "/vertex.CSR";
    string el = outpath + "/edge.CSR";
    ofstream ofs;
    ofs.open(vl.c_str());
    if (ofs.is_open())
    {
        ofs.write((char*)&(vertexlist[0]), sizeof(uint64_t)*vertexlist.size());
    }
    ofs.close();
    ofs.open(el.c_str());
    if (ofs.is_open())
    {
        ofs.write((char*)&(edgelist[0]), sizeof(uint64_t)*edgelist.size());
    }
    ofs.close();
    //================================================//
    t2 = timer::get_usec();
    
    cout<<"== write time: "<<t2-t1<<" sec"<<endl;


    cout<<"==================================================================\n";
    return 0;
}  // end main

