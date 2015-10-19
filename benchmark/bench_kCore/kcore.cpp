//====== Graph Benchmark Suites ======//
//======== kCore Decomposition =======//
//
// Usage: ./kcore.exe --dataset <dataset path> --kcore <k value>

#include "common.h"
#include "def.h"
#include "openG.h"
#include <queue>
#include "omp.h"

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
void arg_init(argument_parser & arg)
{
    arg.add_arg("kcore","3","kCore k value");
}
//==============================================================//
inline unsigned vertex_distributor(uint64_t vid, unsigned threadnum)
{
    return vid%threadnum;
}

void seq_init(graph_t& g, size_t k, size_t & remove_cnt, queue<vertex_iterator> process_q)
{
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
}
size_t kcore(graph_t& g, size_t k, size_t remove_cnt, queue<vertex_iterator> process_q,
        gBenchPerf_event & perf, int perf_group) 
{
    perf.open(perf_group);
    perf.start(perf_group);

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

    perf.stop(perf_group);
    return remove_cnt;
}  // end kcore
void parallel_init(graph_t& g, size_t k, unsigned threadnum, size_t & remove_cnt,
        vector<vector<uint64_t> >& global_input_tasks)
{
    global_input_tasks.resize(threadnum);
    for (vertex_iterator vit=g.vertices_begin(); vit!=g.vertices_end(); vit++) 
    {
        size_t degree = vit->edges_size();
        vit->property().degree = degree;

        if (degree < k) 
        {
            global_input_tasks[vertex_distributor(vit->id(), threadnum)].push_back(vit->id());
            vit->property().removed = true;
            remove_cnt++;
        }
    }
}
size_t parallel_kcore(graph_t& g, size_t k, unsigned threadnum, size_t remove_cnt,
        vector<vector<uint64_t> >& global_input_tasks,
        gBenchPerf_multi & perf, int perf_group)
{
    vector<vector<uint64_t> > global_output_tasks(threadnum*threadnum);
    bool stop = false;
    #pragma omp parallel num_threads(threadnum) shared(stop,global_input_tasks,global_output_tasks) 
    {
        unsigned tid = omp_get_thread_num();
        vector<uint64_t> & input_tasks = global_input_tasks[tid];
        
        perf.open(tid, perf_group);
        perf.start(tid, perf_group);  
        while(!stop)
        {
            #pragma omp barrier
            // process local queue
            stop = true;
            
        
            for (unsigned i=0;i<input_tasks.size();i++)
            {
                uint64_t vid=input_tasks[i];
                vertex_iterator vit = g.find_vertex(vid);
                
                for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++)
                {
                    uint64_t dest_vid = eit->target();
                    vertex_iterator destvit = g.find_vertex(dest_vid);

                    if (destvit->property().removed==false)
                    {
                        unsigned oldval = __sync_fetch_and_sub(&(destvit->property().degree), 1);
                        if (oldval == k)
                        {
                            destvit->property().removed=true;
                            __sync_fetch_and_add(&remove_cnt, 1);
                            global_output_tasks[vertex_distributor(dest_vid,threadnum)+tid*threadnum].push_back(dest_vid);
                        }
                    }
                }
            }
            #pragma omp barrier
            input_tasks.clear();
            for (unsigned i=0;i<threadnum;i++)
            {
                if (global_output_tasks[i*threadnum+tid].size()!=0)
                {
                    stop = false;
                    input_tasks.insert(input_tasks.end(),
                            global_output_tasks[i*threadnum+tid].begin(),
                            global_output_tasks[i*threadnum+tid].end());
                    global_output_tasks[i*threadnum+tid].clear();
                }
            }
            #pragma omp barrier

        }
        perf.stop(tid, perf_group);
    }

    return remove_cnt;
}
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
void reset_graph(graph_t & g)
{
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        vit->property().degree = 0;
        vit->property().removed = false;
    }

}

//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: kCore decomposition\n";
    double t1, t2;

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

    size_t k,threadnum;
    arg.get_value("kcore",k);
    arg.get_value("threadnum",threadnum);
    
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
    if (graph.load_csv_edges(path, true, separator, 0, 1) == -1)
        return -1;
#endif

    size_t vertex_num = graph.num_vertices();
    size_t edge_num = graph.num_edges();
    t2 = timer::get_usec();

    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
    
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n\n";
#endif

    cout<<"computing kCore: k="<<k<<"\n";
    size_t remove_cnt;

    gBenchPerf_multi perf_multi(threadnum, perf);
    unsigned run_num = ceil(perf.get_event_cnt() /(double) DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;
    double elapse_time = 0;
    
    for (unsigned i=0;i<run_num;i++)
    {
        remove_cnt = 0;
        queue<vertex_iterator> process_q;
        vector<vector<uint64_t> > global_input_tasks(threadnum);

        if (threadnum==1)
            seq_init(graph,k,remove_cnt,process_q);
        else
            parallel_init(graph,k,threadnum,remove_cnt,global_input_tasks);

        t1 = timer::get_usec();

        if (threadnum==1)
            remove_cnt=kcore(graph, k,remove_cnt, process_q,perf, i);
        else
            remove_cnt=parallel_kcore(graph, k, threadnum,remove_cnt, global_input_tasks,perf_multi, i);
        t2 = timer::get_usec();
        elapse_time += t2-t1;
        if ((i+1)<run_num) reset_graph(graph);
    }
    cout<<"== removed vertices: "<<remove_cnt<<endl;
#ifndef ENABLE_VERIFY
    cout<<"== time: "<<elapse_time/run_num<<" sec\n";
    if (threadnum == 1)
        perf.print();
    else
        perf_multi.print();

#endif

#ifdef ENABLE_OUTPUT
    cout<<"\n";
    output(graph);
#endif
    cout<<"==================================================================\n";
    return 0;
}  // end main

