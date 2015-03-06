//====== Graph Benchmark Suites ======//
//======== Connected Component =======//
//
// Usage: ./tc.exe --dataset <dataset path>

#include "../lib/common.h"
#include "../lib/def.h"
#include "openG.h"
#include "omp.h"
#include <set>
#include <vector>

using namespace std;

class vertex_property
{
public:
    vertex_property():count(0){}

    uint64_t count;
    std::set<uint64_t> neighbor_set;
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
    unsigned threadnum;
};

void arg_init(arg_t& arguments)
{
    arguments.dataset_path.clear();
    arguments.threadnum=1;
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
        else if (inputarg[i]=="--threadnum")
        {
            i++;
            arguments.threadnum=atol(inputarg[i].c_str());
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
size_t get_intersect_cnt(set<size_t>& setA, set<size_t>& setB)
{
    size_t ret=0;
    set<uint64_t>::iterator iter1=setA.begin(), iter2=setB.begin();

    while (iter1!=setA.end() && iter2!=setB.end()) 
    {
        if ((*iter1) < (*iter2)) 
            iter1++;
        else if ((*iter1) > (*iter2)) 
            iter2++;
        else
        {
            ret++;
            iter1++;
            iter2++;
        }
    }

    return ret;
}



size_t triangle_count(graph_t& g)
{
    size_t ret=0;

    // prepare neighbor set for each vertex
    for (vertex_iterator vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        vit->property().count = 0;
        set<uint64_t> & cur_set = vit->property().neighbor_set;
        for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++) 
        {
            cur_set.insert(eit->target());
        }
    }

    // run triangle count now
    for (vertex_iterator vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        set<uint64_t> & src_set = vit->property().neighbor_set;

        for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++) 
        {
            if (vit->id() > eit->target()) continue; // skip reverse edges
            vertex_iterator vit_targ = g.find_vertex(eit->target());

            set<uint64_t> & dest_set = vit_targ->property().neighbor_set;
            size_t cnt = get_intersect_cnt(src_set, dest_set);

            vit->property().count += cnt;
            vit_targ->property().count += cnt;
        }
    }

    // tune the per-vertex count
    for (vertex_iterator vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        vit->property().count /= 2;
        ret += vit->property().count;
    }

    ret /= 3;

    return ret;
}
size_t parallel_triangle_count(graph_t& g, unsigned threadnum)
{
    size_t ret=0;
    uint64_t chunk = (unsigned)ceil(g.num_vertices()/(double)threadnum);
    #pragma omp parallel num_threads(threadnum)
    {
        unsigned tid = omp_get_thread_num();

        unsigned start = tid*chunk;
        unsigned end = start + chunk;
        if (end > g.num_vertices()) end = g.num_vertices();

        // prepare neighbor set for each vertex        
        for (uint64_t vid=start;vid<end;vid++)
        {
            vertex_iterator vit = g.find_vertex(vid);
            
            vit->property().count = 0;
            set<uint64_t> & cur_set = vit->property().neighbor_set;
            for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++) 
            {
                cur_set.insert(eit->target());
            }
        }
        #pragma omp barrier
        // run triangle count now
        for (uint64_t vid=start;vid<end;vid++)
        {
            vertex_iterator vit = g.find_vertex(vid);

            set<uint64_t> & src_set = vit->property().neighbor_set;

            for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++) 
            {
                if (vit->id() > eit->target()) continue; // skip reverse edges
                vertex_iterator vit_targ = g.find_vertex(eit->target());

                set<uint64_t> & dest_set = vit_targ->property().neighbor_set;
                size_t cnt = get_intersect_cnt(src_set, dest_set);

                __sync_fetch_and_add(&(vit->property().count), cnt);
                __sync_fetch_and_add(&(vit_targ->property().count), cnt);
            }
        }
        #pragma omp barrier 
        // tune the per-vertex count
        for (uint64_t vid=start;vid<end;vid++)
        {
            vertex_iterator vit = g.find_vertex(vid);
            vit->property().count /= 2;
            __sync_fetch_and_add(&ret, vit->property().count);
        }
    }


    ret /= 3;

    return ret;
}

void output(graph_t& g)
{
    cout<<"Triangle Count Results: \n";
    for (vertex_iterator vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        cout<<"== vertex "<<vit->id()<<": count "<<vit->property().count<<endl;
    }
}


int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: triangle count\n";

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

    uint64_t vertex_num = graph.num_vertices();
    uint64_t edge_num = graph.num_edges();
    t2 = timer::get_usec();
    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

    cout<<"\ncomputing triangle count...\n";
    size_t tcount;
    t1 = timer::get_usec();
    perf.start();

    if (arguments.threadnum==1)
        tcount = triangle_count(graph);
    else
        tcount = parallel_triangle_count(graph, arguments.threadnum);

    perf.stop();
    t2 = timer::get_usec();
    cout<<"== total triangle count: "<<tcount<<endl;
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

