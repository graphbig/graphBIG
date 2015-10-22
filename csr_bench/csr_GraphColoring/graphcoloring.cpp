#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>
#include <algorithm>
#include "common.h"

#ifdef USE_OMP
#include "omp.h"
#endif

#include "pthread.h"

#ifdef HMC
#include "HMC.h"
#endif

#ifdef SIM
#include "SIM.h"
#endif

using namespace std;

#define SEED 123

#define MY_INFINITY 0xfff0

pthread_barrier_t barrier;

inline unsigned vertex_distributor(uint64_t vid, unsigned threadnum)
{
    return vid%threadnum;
}
void seq_graph_coloring(
        uint64_t * vertexlist, 
        uint64_t * edgelist, uint16_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt)
{
    double t1, t2;
       
    t1 = timer::get_usec();

    // initializzation
    srand(SEED);
    vector<uint16_t> vertex_rand(vertex_cnt);
    vector<uint64_t> task1, task2;
    vector<uint64_t> * input_tasks=&task1;
    vector<uint64_t> * output_tasks=&task2;
    input_tasks->resize(vertex_cnt);
    for (unsigned i=0; i<vertex_cnt; i++)
    {
        vproplist[i] = MY_INFINITY;
        vertex_rand[i] = rand();
        (*input_tasks)[i] = i;
    }
    
    t2 = timer::get_usec();
    cout<<"== initialization time: "<<t2-t1<<" sec\n";

    unsigned color = 0;
    while(input_tasks->size())
    {
        for (unsigned i=0;i<input_tasks->size();i++)
        {
            uint64_t vid = (*input_tasks)[i];
            
            uint16_t local_rand = vertex_rand[vid];

            unsigned start = vertexlist[vid];
            unsigned end = vertexlist[vid+1];
            bool found_larger = false;
            for (unsigned d=start;d<end;d++)
            {
                uint64_t dest = edgelist[d];
                if (vproplist[dest]<color) continue;
                if ( (vertex_rand[dest]>local_rand) ||
                        (vertex_rand[dest]==local_rand && dest<vid))
                {
                    found_larger = true;
                    break;
                }
            }
            if (found_larger == false)
                vproplist[vid] = color;
            else
                output_tasks->push_back(vid);
        }

        input_tasks->clear();
        swap(input_tasks, output_tasks);
        color++;
    }
}
struct arg_t
{
    uint64_t * vertexlist; 
    uint64_t * edgelist; 
    uint16_t * vproplist;
    uint64_t vertex_cnt;
    uint64_t edge_cnt;

    vector<vector<uint64_t> > * global_input_tasks_ptr;
    vector<vector<uint64_t> > * global_output_tasks_ptr;
    vector<uint16_t>  * vertex_rand_ptr;
    unsigned tid;
    unsigned threadnum;
    bool * stop;
};
void* thread_work(void * t)
{
    struct arg_t * arg = (struct arg_t *) t;
    vector<vector<uint64_t> > & global_input_tasks = *(arg->global_input_tasks_ptr);
    vector<vector<uint64_t> > & global_output_tasks = *(arg->global_output_tasks_ptr);
    vector<uint16_t> & vertex_rand = *(arg->vertex_rand_ptr);

    uint64_t * vertexlist = arg->vertexlist; 
    uint64_t * edgelist = arg->edgelist; 
    uint16_t * vproplist = arg->vproplist;
    unsigned tid = arg->tid;
    bool & stop = *(arg->stop);
    unsigned threadnum = arg->threadnum;

    vector<uint64_t> & input_tasks = global_input_tasks[tid];
#ifdef SIM
    pthread_barrier_wait (&barrier);
    SIM_BEGIN(true);
#endif  
    vector<uint16_t> updatelist;
    unsigned color=0; 
    while(!stop)
    {
        pthread_barrier_wait (&barrier);
        // process local queue
        stop = true;
                    
        for (unsigned i=0;i<input_tasks.size();i++)
        {
            uint64_t vid=input_tasks[i];
            uint16_t local_rand = vertex_rand[vid];

            unsigned start = vertexlist[vid];
            unsigned end = vertexlist[vid+1];
            bool found_larger = false;
            for (unsigned d=start;d<end;d++)
            {
                uint64_t dest = edgelist[d];
                if (vproplist[dest]<color) continue;
                if ( (vertex_rand[dest]>local_rand) ||
                    (vertex_rand[dest]==local_rand && dest<vid))
                {
                    found_larger = true;
                    break;
                }
            }
            // if local vertex has max num, color it
            if (found_larger == false)
                vproplist[vid]=color;//updatelist.push_back(vid);
            else // otherwise, need to processed again 
                global_output_tasks[vertex_distributor(vid,threadnum)+tid*threadnum].push_back(vid);
        }
        pthread_barrier_wait (&barrier);
        //for (unsigned i=0;i<updatelist.size();i++)
        //{
        //    vproplist[updatelist[i]] = color;
        //}
        //updatelist.clear();
        input_tasks.clear();
        // generate input worklists from output worklists
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
        
        color++;
        pthread_barrier_wait (&barrier);
    }
#ifdef SIM
    SIM_END(true);
#endif 

    if (tid!=0) pthread_exit((void*) t);

    return NULL; 
}
void parallel_graph_coloring(
        uint64_t * vertexlist, 
        uint64_t * edgelist, uint16_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt,
        unsigned threadnum)
{
    double t1, t2;
       
    t1 = timer::get_usec();

    // initializzation
    srand(SEED);
    vector<uint16_t> vertex_rand(vertex_cnt);
    vector<vector<uint64_t> > global_input_tasks(threadnum);
    vector<vector<uint64_t> > global_output_tasks(threadnum*threadnum);

    for (unsigned i=0; i<vertex_cnt; i++)
    {
        vproplist[i] = MY_INFINITY;
        vertex_rand[i] = rand();
        global_input_tasks[vertex_distributor(i, threadnum)].push_back(i);
    }
    
    t2 = timer::get_usec();
    cout<<"== initialization time: "<<t2-t1<<" sec\n";

    t1 = timer::get_usec();
    
    bool stop = false;
#ifndef USE_OMP
    pthread_barrier_init (&barrier, NULL, threadnum);

    pthread_t thread[threadnum];
    pthread_attr_t attr; 
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    
    struct arg_t args[threadnum];
    for (unsigned t=0;t<threadnum;t++)
    {
        args[t].vertexlist = vertexlist; 
        args[t].edgelist = edgelist; 
        args[t].vproplist = vproplist;
        args[t].vertex_cnt = vertex_cnt;
        args[t].edge_cnt = edge_cnt;

        args[t].global_input_tasks_ptr = &global_input_tasks;
        args[t].global_output_tasks_ptr = &global_output_tasks;
        args[t].vertex_rand_ptr = &vertex_rand;

        args[t].tid = t;
        args[t].stop = &stop;
        args[t].threadnum = threadnum;
    }

    for(unsigned t=1; t<threadnum; t++) 
    {
        int rc = pthread_create(&thread[t], &attr, thread_work, (&(args[t]))); 
        if (rc) 
        {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    thread_work((void*) &(args[0]));

    pthread_attr_destroy(&attr);
    for(unsigned t=1; t<threadnum; t++) 
    {
        void* status;
        int rc = pthread_join(thread[t], &status);
        if (rc) 
        {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
    }
        
#else    
    #pragma omp parallel num_threads(threadnum) shared(stop,global_input_tasks,global_output_tasks) 
    {
        unsigned tid = omp_get_thread_num();
        vector<uint64_t> & input_tasks = global_input_tasks[tid];
        vector<uint16_t> updatelist;
        unsigned color=0; 
        while(!stop)
        {
            #pragma omp barrier
            // process local queue
            stop = true;
                        
            for (unsigned i=0;i<input_tasks.size();i++)
            {
                uint64_t vid=input_tasks[i];
                uint16_t local_rand = vertex_rand[vid];

                unsigned start = vertexlist[vid];
                unsigned end = vertexlist[vid+1];
                bool found_larger = false;
                for (unsigned d=start;d<end;d++)
                {
                    uint64_t dest = edgelist[d];
                    if (vproplist[dest]<color) continue;
                    if ( (vertex_rand[dest]>local_rand) ||
                        (vertex_rand[dest]==local_rand && dest<vid))
                    {
                        found_larger = true;
                        break;
                    }
                }
                // if local vertex has max num, color it
                if (found_larger == false)
                    vproplist[vid]=color;//updatelist.push_back(vid);
                else // otherwise, need to processed again 
                    global_output_tasks[vertex_distributor(vid,threadnum)+tid*threadnum].push_back(vid);
            }
            #pragma omp barrier
            //for (unsigned i=0;i<updatelist.size();i++)
            //{
            //    vproplist[updatelist[i]] = color;
            //}
            //updatelist.clear();
            input_tasks.clear();
            // generate input worklists from output worklists
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
            
            color++;
            #pragma omp barrier
        }
    }
#endif
    t2 = timer::get_usec();
    cout<<"== traversal time: "<<t2-t1<<" sec\n";
}

