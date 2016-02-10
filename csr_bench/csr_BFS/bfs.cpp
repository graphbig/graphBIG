#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <stdint.h>
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

#define MY_INFINITY 0xfff0

pthread_barrier_t   barrier;

inline unsigned vertex_distributor(uint64_t vid, unsigned threadnum)
{
    return vid%threadnum;
}

void seq_BFS(
        uint64_t * vertexlist, 
        uint64_t * edgelist, uint16_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt,
        uint64_t root)
{
    double t1, t2;
    
    t1 = timer::get_usec();

    // initializzation
    for (unsigned i=0; i<vertex_cnt; i++)
    {
        vproplist[i] = MY_INFINITY;
    }
    vproplist[root] = 0;

    std::queue<uint64_t> vertex_queue;
    vertex_queue.push(root);
    
    t2 = timer::get_usec();

#ifndef ENABLE_VERIFY
    cout<<"== initialization time: "<<t2-t1<<" sec\n";
#else
    (void)t1;
    (void)t2;
#endif

    t1 = timer::get_usec();
#ifdef SIM
        SIM_BEGIN(true);
#endif           
        
    while(!vertex_queue.empty())
    {
        uint64_t vid=vertex_queue.front();
        vertex_queue.pop();

        uint16_t curr_level = vproplist[vid];
        uint32_t edge_start = vertexlist[vid];
        uint32_t edge_end = vertexlist[vid+1];

        for (unsigned j=edge_start; j<edge_end; j++)
        {
            uint64_t dest_vid = edgelist[j];
#ifdef HMC
            if (HMC_CAS_equal_16B(&(vproplist[dest_vid]),
                MY_INFINITY,curr_level+1) == (curr_level+1))
            {
#else
            if (vproplist[dest_vid]==MY_INFINITY)
            {
                vproplist[dest_vid] = curr_level+1;
#endif                
                vertex_queue.push(dest_vid);
            }
        }
    }
#ifdef SIM
        SIM_END(true);
#endif

    t2 = timer::get_usec();
#ifndef ENABLE_VERIFY
    cout<<"== traversal time: "<<t2-t1<<" sec\n";
#endif
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
    unsigned tid;
    unsigned threadnum;
    bool * stop;
};

void* thread_work(void * t)
{
    struct arg_t * arg = (struct arg_t *) t;
    vector<vector<uint64_t> > & global_input_tasks = *(arg->global_input_tasks_ptr);
    vector<vector<uint64_t> > & global_output_tasks = *(arg->global_output_tasks_ptr);
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
    while(!stop)
    {
        pthread_barrier_wait (&barrier);
        // process local queue
        stop = true;
        
    
        for (unsigned i=0;i<input_tasks.size();i++)
        {
            uint64_t vid=input_tasks[i];
            uint16_t curr_level = vproplist[vid];
            uint32_t edge_start = vertexlist[vid];
            uint32_t edge_end = vertexlist[vid+1];

            for (unsigned j=edge_start; j<edge_end; j++)
            {
                uint64_t dest_vid = edgelist[j];
#ifdef HMC
                if (HMC_CAS_equal_16B(&(vproplist[dest_vid]),
                            MY_INFINITY,curr_level+1) == MY_INFINITY)
#else
                if (__sync_bool_compare_and_swap(&(vproplist[dest_vid]), 
                            MY_INFINITY,curr_level+1))
#endif
                {
                    global_output_tasks[vertex_distributor(dest_vid,threadnum)+tid*threadnum].push_back(dest_vid);
                }
            }
        }
        pthread_barrier_wait (&barrier);
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
        pthread_barrier_wait (&barrier);

    }

#ifdef SIM
    SIM_END(true);
#endif    
    
    if (tid!=0) pthread_exit((void*) t);

    return NULL;
}

void parallel_BFS(
        uint64_t * vertexlist, 
        uint64_t * edgelist, uint16_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt,
        uint64_t root, unsigned threadnum)
{
    double t1, t2;
    
    t1 = timer::get_usec();

    // initializzation
    for (unsigned i=0; i<vertex_cnt; i++)
    {
        vproplist[i] = MY_INFINITY;
    }
    vproplist[root] = 0;

    vector<vector<uint64_t> > global_input_tasks(threadnum);
    global_input_tasks[vertex_distributor(root,threadnum)].push_back(root);
    
    vector<vector<uint64_t> > global_output_tasks(threadnum*threadnum);
    t2 = timer::get_usec();
#ifndef ENABLE_VERIFY
    cout<<"== initialization time: "<<t2-t1<<" sec\n";
#else
    (void)t1;
    (void)t2;
#endif
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
#ifdef SIM
        SIM_BEGIN(true);
#endif        
        while(!stop)
        {
            #pragma omp barrier
            // process local queue
            stop = true;
            
        
            for (unsigned i=0;i<input_tasks.size();i++)
            {
                uint64_t vid=input_tasks[i];
                uint16_t curr_level = vproplist[vid];
                uint32_t edge_start = vertexlist[vid];
                uint32_t edge_end = vertexlist[vid+1];

                for (unsigned j=edge_start; j<edge_end; j++)
                {
                    uint64_t dest_vid = edgelist[j];
#ifdef HMC
                    if (HMC_CAS_equal_16B(&(vproplist[dest_vid]),
                                MY_INFINITY,curr_level+1) == (curr_level+1))
#else
                    if (__sync_bool_compare_and_swap(&(vproplist[dest_vid]), 
                                MY_INFINITY,curr_level+1))
#endif
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
#ifdef SIM
        SIM_END(true);
#endif
    }
#endif
    t2 = timer::get_usec();
#ifndef ENABLE_VERIFY
    cout<<"== traversal time: "<<t2-t1<<" sec\n";
#endif
}

