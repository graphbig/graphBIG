//====== Graph Benchmark Suites ======//
//======= Breadth-first Search =======//
//
// Usage: ./bfs.exe --dataset <dataset path> --root <root vertex id>

#include "common.h"
#include "def.h"
#include "perf.h"

#include "openG.h"
#include <queue>
#include "omp.h"

#ifdef SIM
#include "SIM.h"
#endif

using namespace std;

#define MY_INFINITY 0xffffff00

class vertex_property
{
public:
    vertex_property():color(COLOR_WHITE),order(0),level(MY_INFINITY){}
    vertex_property(uint8_t x):color(x),order(0),level(MY_INFINITY){}

    uint8_t color;
    uint64_t order;
    uint64_t level;
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

class BFSVisitor
{
public:
    void white_vertex(vertex_iterator vit){white_access++;}
    void grey_vertex(vertex_iterator vit){grey_access++;}
    void black_vertex(vertex_iterator vit){black_access++;}
    void finish_vertex(vertex_iterator vit){}

    size_t white_access;
    size_t grey_access;
    size_t black_access;

    BFSVisitor()
    {
        white_access=0;
        grey_access=0;
        black_access=0;
    }
};
inline unsigned vertex_distributor(uint64_t vid, unsigned threadnum)
{
    return vid%threadnum;
}
void parallel_bfs(graph_t& g, size_t root, unsigned threadnum, gBenchPerf_multi & perf, int perf_group)
{
    // initializzation
    vertex_iterator rootvit=g.find_vertex(root);
    if (rootvit==g.vertices_end()) return;

    rootvit->property().level = 0;

    vector<vector<uint64_t> > global_input_tasks(threadnum);
    global_input_tasks[vertex_distributor(root, threadnum)].push_back(root);
    
    vector<vector<uint64_t> > global_output_tasks(threadnum*threadnum);

    bool stop = false;
    #pragma omp parallel num_threads(threadnum) shared(stop,global_input_tasks,global_output_tasks,perf) 
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
                uint32_t curr_level = vit->property().level;
                
                for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++)
                {
                    uint64_t dest_vid = eit->target();
                    vertex_iterator destvit = g.find_vertex(dest_vid);
                    if (__sync_bool_compare_and_swap(&(destvit->property().level), 
                                MY_INFINITY,curr_level+1))
                    {
                        global_output_tasks[vertex_distributor(dest_vid,threadnum)+tid*threadnum].push_back(dest_vid);
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

}

void bfs(graph_t& g, size_t root, BFSVisitor& vis, gBenchPerf_event & perf, int perf_group) 
{
    perf.open(perf_group);
    perf.start(perf_group);

    std::queue<vertex_iterator> vertex_queue;

    vertex_iterator iter = g.find_vertex(root);
    if (iter == g.vertices_end()) 
        return;

    vis.white_vertex(iter);
    size_t visit_cnt=0;

    iter->property().color = COLOR_GREY;
    iter->property().order = 0;
    iter->property().level = 0;

    vertex_queue.push(iter);
    visit_cnt++;
#ifdef SIM
    SIM_BEGIN(true);
#endif
    while (!vertex_queue.empty()) 
    {
        vertex_iterator u = vertex_queue.front(); 
        vertex_queue.pop();  

        for (edge_iterator ei = u->edges_begin(); ei != u->edges_end(); ++ei) 
        {
            vertex_iterator v = g.find_vertex(ei->target()); 


            uint8_t v_color = v->property().color;

            if (v_color == COLOR_WHITE) 
            {
                vis.white_vertex(v);

                v->property().color = COLOR_GREY;
                v->property().order = visit_cnt;
                v->property().level = u->property().level+1;

                vertex_queue.push(v);
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

}  // end bfs
//==============================================================//

void output(graph_t& g)
{
    cout<<"BFS Results: \n";
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        cout<<"== vertex "<<vit->id()<<": level "<<vit->property().level<<"\n";
    }
}

void reset_graph(graph_t & g)
{
    vertex_iterator vit;
    for (vit=g.vertices_begin(); vit!=g.vertices_end(); vit++)
    {
        vit->property().color = COLOR_WHITE;
        vit->property().order = 0;
        vit->property().level = MY_INFINITY;
    }

}

//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Benchmark: BFS\n";

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

    size_t root,threadnum;
    arg.get_value("root",root);
    arg.get_value("threadnum",threadnum);
    
    graph_t graph;
    double t1, t2;

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

    size_t vertex_num = graph.vertex_num();
    size_t edge_num = graph.edge_num();
    t2 = timer::get_usec();
    cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";

#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

    BFSVisitor vis;

    cout<<"\nBFS root: "<<root<<"\n";
    
    gBenchPerf_multi perf_multi(threadnum, perf);
    unsigned run_num = ceil(perf.get_event_cnt() /(double) DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;
    double elapse_time = 0;
    
    for (unsigned i=0;i<run_num;i++)
    {
        t1 = timer::get_usec();

        if (threadnum==1)
            bfs(graph, root, vis, perf, i);
        else
            parallel_bfs(graph, root, threadnum, perf_multi, i);

        t2 = timer::get_usec();
        elapse_time += t2-t1;
        if ((i+1)<run_num) reset_graph(graph);
    }
    cout<<"BFS finish: \n";

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

    cout<<"=================================================================="<<endl;
    return 0;
}  // end main

