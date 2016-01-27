//====== Graph Benchmark Suites ======//
//======== Connected Component =======//
//
// Usage: ./tc.exe --dataset <dataset path>

#include "common.h"
#include "def.h"
#include "openG.h"
#include "omp.h"
#include <set>
#include <vector>
#include <algorithm>

#ifdef HMC
#include "HMC.h"
#endif

#ifdef SIM
#include "SIM.h"
#endif

using namespace std;

size_t maxiter = 0;
size_t beginiter = 0;
size_t enditer = 0;

class vertex_property
{
public:
    vertex_property():count(0){}

    int16_t count;
    std::vector<uint64_t> neighbor_set;
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
    arg.add_arg("maxiter","0","maximum loop iteration (0-unlimited, only set for simulation purpose)");
}
//==============================================================//
size_t get_intersect_cnt(vector<size_t>& setA, vector<size_t>& setB)
{
    size_t ret=0;
    vector<uint64_t>::iterator iter1=setA.begin(), iter2=setB.begin();

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

void tc_init(graph_t& g)
{
    // prepare neighbor set for each vertex
    for (vertex_iterator vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        vit->property().count = 0;
        vector<uint64_t> & cur_set = vit->property().neighbor_set;
        cur_set.reserve(vit->edges_size());
        for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++) 
        {
            cur_set.push_back(eit->target());
        }
        std::sort(cur_set.begin(),cur_set.end());
    }
}

size_t triangle_count(graph_t& g, gBenchPerf_event & perf, int perf_group)
{
    perf.open(perf_group);
    perf.start(perf_group);

    size_t ret=0;

    // run triangle count now
    for (vertex_iterator vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        vector<uint64_t> & src_set = vit->property().neighbor_set;

        for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++) 
        {
            if (vit->id() > eit->target()) continue; // skip reverse edges
            vertex_iterator vit_targ = g.find_vertex(eit->target());

            vector<uint64_t> & dest_set = vit_targ->property().neighbor_set;
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

    perf.stop(perf_group);
    return ret;
}

void gen_workset(graph_t& g, vector<unsigned>& workset, unsigned threadnum)
{
    unsigned chunk = (unsigned)ceil(g.num_edges()/(double)threadnum);
    unsigned last=0, curr=0, th=1;
    workset.clear();
    workset.resize(threadnum+1,0);
    for (vertex_iterator vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        curr += vit->edges_size();
        if ((curr-last)>=chunk)
        {
            last = curr;
            workset[th] = vit->id();
            if (th<threadnum) th++;
        }
    }
    workset[threadnum] = g.num_vertices();
    //for (unsigned i=0;i<=threadnum;i++)
    //    cout<<workset[i]<<" ";
    //cout<<endl;
}

void parallel_tc_init(graph_t& g, unsigned threadnum)
{
    vector<unsigned> ws;
    gen_workset(g, ws, threadnum);

    #pragma omp parallel num_threads(threadnum)
    {
        unsigned tid = omp_get_thread_num();

        // prepare neighbor set for each vertex        
        for (uint64_t vid=ws[tid];vid<ws[tid+1];vid++)
        {
            vertex_iterator vit = g.find_vertex(vid);
            if (vit == g.vertices_end()) continue;

            vit->property().count = 0;
            vector<uint64_t> & cur_set = vit->property().neighbor_set;
            cur_set.reserve(vit->edges_size());
            for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++) 
            {
                cur_set.push_back(eit->target());
            }
            std::sort(cur_set.begin(),cur_set.end());
        }
    }
}

void parallel_workset_init(graph_t&g, vector<unsigned>& workset, unsigned threadnum)
{
    vector<unsigned> n_op(g.num_vertices(),0);
    unsigned totalcnt=0;
    vector<unsigned> ws;
    gen_workset(g, ws, threadnum);

    #pragma omp parallel num_threads(threadnum)
    {
        unsigned tid = omp_get_thread_num();
   
        unsigned tot=0; 
        for (uint64_t vid=ws[tid];vid<ws[tid+1];vid++)
        {
            vertex_iterator vit = g.find_vertex(vid);
            if (vit == g.vertices_end()) continue;

            for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++) 
            {
                vertex_iterator vit_targ = g.find_vertex(eit->target());
                n_op[vid] += vit->edges_size() + vit_targ->edges_size();
            }
            tot += n_op[vid];
        }
        __sync_fetch_and_add(&(totalcnt),tot);
    }
    workset.clear();
    workset.resize(threadnum+1,0);
    unsigned last=0, cnt=0, th=1;
    unsigned cnt_chunk = (unsigned)ceil(totalcnt/(double)threadnum);
    for (uint64_t vid=0; vid<g.num_vertices();vid++)
    {
        cnt += n_op[vid];
        if ((cnt-last) > cnt_chunk)
        {
            if (th<threadnum) workset[th] = vid;
            th++;
            last = cnt;
        }
    }
    workset[threadnum] = g.num_vertices();
    
}

size_t parallel_triangle_count(graph_t& g, unsigned threadnum, vector<unsigned>& workset, 
        gBenchPerf_multi & perf, int perf_group)
{
    size_t ret=0;
    
    #pragma omp parallel num_threads(threadnum)
    {
        unsigned tid = omp_get_thread_num();

        perf.open(tid, perf_group);
        perf.start(tid, perf_group);  
        unsigned start = workset[tid];
        unsigned end = workset[tid+1];
        if (maxiter != 0 && (start+maxiter) < end)
            end = start + maxiter;
        if (end > g.num_vertices()) end = g.num_vertices();
        
        // for test only
        //if (end > (start+1000)) end = start+1000;
#ifdef SIM
        unsigned iter = 0;
#endif
        // run triangle count now
        for (uint64_t vid=start;vid<end;vid++)
        {
#ifdef SIM
            SIM_BEGIN(iter==beginiter);
            iter++;
#endif
            vertex_iterator vit = g.find_vertex(vid);

            vector<uint64_t> & src_set = vit->property().neighbor_set;

            for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++) 
            {
                if (vit->id() > eit->target()) continue; // skip reverse edges
                vertex_iterator vit_targ = g.find_vertex(eit->target());

                vector<uint64_t> & dest_set = vit_targ->property().neighbor_set;
                size_t cnt = get_intersect_cnt(src_set, dest_set);
#ifdef HMC
                HMC_ADD_16B(&(vit->property().count),cnt);
                HMC_ADD_16B(&(vit_targ->property().count),cnt);
#else   
                __sync_fetch_and_add(&(vit->property().count), cnt);
                __sync_fetch_and_add(&(vit_targ->property().count), cnt);
#endif
            }
#ifdef SIM
            SIM_END(iter==enditer);
#endif
        }
        #pragma omp barrier 
        // tune the per-vertex count
        for (uint64_t vid=start;vid<end;vid++)
        {
            vertex_iterator vit = g.find_vertex(vid);
            vit->property().count /= 2;
            __sync_fetch_and_add(&ret, vit->property().count);
        }
#ifdef SIM
        SIM_END(enditer==0);
#endif  
        perf.stop(tid, perf_group);
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

void reset_graph(graph_t & g)
{
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        vit->property().count = 0;
    }

}

int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: triangle count\n";

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

    size_t threadnum;
    arg.get_value("threadnum",threadnum);
    arg.get_value("maxiter",maxiter);
#ifdef SIM
    arg.get_value("beginiter",beginiter);
    arg.get_value("enditer",enditer);
#endif

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

    uint64_t vertex_num = graph.num_vertices();
    uint64_t edge_num = graph.num_edges();
    t2 = timer::get_usec();
    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

    cout<<"\npreparing neighbor sets..."<<endl;
    vector<unsigned> workset;
    if (threadnum==1)
        tc_init(graph);
    else
    {
        parallel_tc_init(graph, threadnum);
        cout<<"preparing workset..."<<endl;
        //parallel_workset_init(graph, workset, arguments.threadnum);
        gen_workset(graph, workset, threadnum);
    }

    if (maxiter != 0) cout<<"\nmax iteration: "<<maxiter;
    cout<<"\ncomputing triangle count..."<<endl;
    size_t tcount;

    gBenchPerf_multi perf_multi(threadnum, perf);
    unsigned run_num = ceil(perf.get_event_cnt() /(double) DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;
    double elapse_time = 0;
    
    for (unsigned i=0;i<run_num;i++)
    {
        t1 = timer::get_usec();

        if (threadnum==1)
            tcount = triangle_count(graph, perf, i);
        else
            tcount = parallel_triangle_count(graph, threadnum, workset, perf_multi, i);
        t2 = timer::get_usec();

        elapse_time += t2 - t1;
        if ((i+1)<run_num) reset_graph(graph);
    }
    cout<<"== total triangle count: "<<tcount<<endl;
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<elapse_time/run_num<<" sec\n";
    if (threadnum == 1)
        perf.print();
    else
        perf_multi.print();
#endif

#ifdef ENABLE_OUTPUT
    cout<<endl;
    output(graph);
#endif
    cout<<"==================================================================\n";
    return 0;
}  // end main

