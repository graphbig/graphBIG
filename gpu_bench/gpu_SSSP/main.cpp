//====== Graph Benchmark Suites ======//
//

#include <vector>
#include <string>
#include <fstream>
#include "../lib/common.h"
#include "../lib/def.h"
#include "openG.h"

using namespace std;

extern void cuda_SSSP(
        uint64_t * vertexlist, 
        uint64_t * edgelist, 
        uint32_t * vproplist,
        uint32_t * eproplist,
        uint64_t vertex_cnt, 
        uint64_t edge_cnt,
        uint64_t root);

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
    uint64_t root;
};

void arg_init(arg_t& arguments)
{
    arguments.dataset_path.clear();
    arguments.root = 0;
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
        else if (inputarg[i]=="--root")
        {
            i++;
            arguments.root = atoll(inputarg[i].c_str());
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

#ifdef EXTERNAL_CSR
void writeback(const string& fn, vector<uint32_t> & vproplist)
{
    ofstream fout;
    fout.open(fn.c_str(),ofstream::binary);
    if (fout.is_open()==false)
    {
        cout<<"cannot open file: "<<fn<<endl;
        return;
    }
    fout.write((char*)&(vproplist[0]), sizeof(uint32_t)*vproplist.size());
    fout.close();
}
#endif


//==============================================================//

void output(vector<uint32_t> & vproplist)
{
    cout<<"SSSP Results:\n";
    for(size_t i=0;i<vproplist.size();i++)
    {
        cout<<"== vertex "<<i<<": level "<<vproplist[i]<<endl;
    }
}

//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: GPU SSSP\n";

    arg_t arguments;
    vector<string> inputarg;
    argument_parser::initialize(argc,argv,inputarg);
    arg_init(arguments);
    arg_parser(arguments,inputarg);

    graph_t g;
    double t1, t2;
    
    cout<<"loading data... \n";

    size_t vertex_num, edge_num;
    vector<uint64_t> vertexlist, edgelist; 
    
    t1 = timer::get_usec();
#ifdef EXTERNAL_CSR
    if (g.load_CSR_Graph(arguments.dataset_path, 
            vertex_num,edge_num,vertexlist,edgelist)==false)
    {
        cout<<"cannot open csr files"<<endl;
        return -1;
    }
#else
    string vfile = arguments.dataset_path + "/vertex.csv";
    string efile = arguments.dataset_path + "/edge.csv";

    if (g.load_csv_vertices(vfile, true, "|,", 0) == -1)
        return -1;
    if (g.load_csv_edges(efile, true, "|,", 0, 1) == -1) 
        return -1;

    vertex_num = g.num_vertices();
    edge_num = g.num_edges();
#endif    
    t2 = timer::get_usec();

    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
    
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

    t1 = timer::get_usec();
    //================================================//
    // prepare compact data for CUDA side
    g.to_CSR_Graph(vertexlist, edgelist);
    t2 = timer::get_usec();

    cout<<"== data conversion time: "<<t2-t1<<" sec\n"<<endl;

    vector<uint32_t> vproplist(vertex_num, 0);
    vector<uint32_t> eproplist(edge_num, 1);
    //================================================//
    

    t1 = timer::get_usec();
    //================================================//
    // call CUDA function 
    cuda_SSSP(&(vertexlist[0]), 
            &(edgelist[0]), &(vproplist[0]),&(eproplist[0]), 
            vertexlist.size()-1, edgelist.size(), arguments.root);
    //================================================//
    t2 = timer::get_usec();
    

    cout<<"\nGPU SSSP finish: \n";
    cout<<"== "<<g.num_vertices()<<" vertices  "<<g.num_edges()<<" edges\n";
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif
#ifdef EXTERNAL_CSR
    string refile = arguments.dataset_path + "/result.array";
    cout<<"\nResult wrote to file: "<<refile<<endl;
    writeback(refile,vproplist);
#endif 
#ifdef ENABLE_OUTPUT
    cout<<"\n";
    output(vproplist);
#endif

    cout<<"==================================================================\n";
    return 0;
}  // end main

