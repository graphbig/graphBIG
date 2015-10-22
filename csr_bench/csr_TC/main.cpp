//====== Graph Benchmark Suites ======//
//

#include <vector>
#include <string>
#include <fstream>
#include <math.h>
#include "common.h"
#include "def.h"
//#include "omp.h"
#include "openG.h"

using namespace std;

extern unsigned seq_triangle_count(
        uint64_t * vertexlist, 
        uint64_t * edgelist, int16_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt); 
extern unsigned parallel_triangle_count(
        uint64_t * vertexlist, 
        uint64_t * edgelist, int16_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt,
        unsigned threadnum);

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

//==============================================================//
void init(vector<uint64_t>& vertexlist, 
        vector<uint64_t>& edgelist,
        unsigned threadnum)
{
    //unsigned chunk = ceil((vertexlist.size()-1)/(double)threadnum);

    for (unsigned vid=0;vid<vertexlist.size()-1;vid++)
    {
        std::sort(edgelist.begin()+vertexlist[vid], 
                    edgelist.begin()+vertexlist[vid+1]);       
    }
/*    
    #pragma omp parallel num_threads(threadnum)
    {
        unsigned tid = omp_get_thread_num();

        unsigned start = tid*chunk;
        unsigned end = start + chunk;
        if (end > (vertexlist.size()-1)) end = vertexlist.size()-1;
    
        for (unsigned vid=start;vid<end;vid++)
        {
            std::sort(edgelist.begin()+vertexlist[vid], 
                    edgelist.begin()+vertexlist[vid+1]);
        }

    }
*/
    return;
}
//==============================================================//

void output(vector<int16_t> & vproplist)
{
    cout<<"Triangle Count Results:\n";
    for(size_t i=0;i<vproplist.size();i++)
    {
        cout<<"== vertex "<<i<<": count "<<vproplist[i]<<endl;
    }
}

//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: Triangle Count\n";

    argument_parser arg;
#ifndef NO_PERF    
    gBenchPerf_event perf;
    if (arg.parse(argc,argv,perf,false)==false)
    {
        arg.help();
        return -1;
    }
#else
    if (arg.parse(argc,argv,false)==false)
    {
        arg.help();
        return -1;
    }
#endif
    string path;
    arg.get_value("dataset",path);

    size_t threadnum;
    arg.get_value("threadnum",threadnum);

    double t1, t2;
    
    cout<<"loading data... \n";

    t1 = timer::get_usec();
    string vfile = path + "/vertex.CSR";
    string efile = path + "/edge.CSR";

    vector<uint64_t> vertexlist, edgelist; 
    size_t vertex_num, edge_num;

    graph_t::load_CSR_Graph(vfile, efile,
            vertex_num, edge_num,
            vertexlist, edgelist);

    t2 = timer::get_usec();

    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
    
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

    //================================================//
    vector<int16_t> vproplist(vertex_num, 0);
    //================================================//
    t1 = timer::get_usec();
    init(vertexlist,edgelist,threadnum);
    t2 = timer::get_usec();
    cout<<"== data preprocessing time: "<<t2-t1<<" sec\n"<<endl;

    unsigned tcount;
    t1 = timer::get_usec();
    //================================================//
    if (threadnum==1)
        tcount = seq_triangle_count(&(vertexlist[0]), 
            &(edgelist[0]), &(vproplist[0]), 
            vertexlist.size()-1, edgelist.size());
    else
        tcount = parallel_triangle_count(&(vertexlist[0]), 
            &(edgelist[0]), &(vproplist[0]), 
            vertexlist.size()-1, edgelist.size(),threadnum);
    //================================================//
    t2 = timer::get_usec();
    

    cout<<"\nTriangle Count finish: \n";
    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
    cout<<"== total triangle count: "<<tcount<<"\n";
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

#ifdef ENABLE_OUTPUT
    cout<<"\n";
    output(vproplist);
#endif

    cout<<"==================================================================\n";
    return 0;
}  // end main

