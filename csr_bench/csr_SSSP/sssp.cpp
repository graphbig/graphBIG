#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>
#include <queue>
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
void seq_SSSP(
        uint64_t * vertexlist, 
        uint64_t * edgelist, 
        uint16_t * vproplist,
        uint16_t * eproplist,
        uint64_t vertex_cnt, 
        uint64_t edge_cnt,
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
    
    std::queue<uint64_t> vertex_q;
    vertex_q.push(root);

    t2 = timer::get_usec();
    cout<<"== initialization time: "<<t2-t1<<" sec\n";


    t1 = timer::get_usec();
#ifdef SIM
    SIM_BEGIN(true);
#endif    
    while(!vertex_q.empty())
    {
            uint64_t vid=vertex_q.front();
            vertex_q.pop();
            
            uint16_t curr_dist = vproplist[vid];
            uint32_t edge_start = vertexlist[vid];
            uint32_t edge_end = vertexlist[vid+1];

            for (unsigned j=edge_start; j<edge_end; j++)
            {
                uint64_t dest_vid = edgelist[j];
                uint16_t new_dist = curr_dist + eproplist[j];
                
                if (vproplist[dest_vid]>new_dist) 
                {
                    vproplist[dest_vid] = new_dist;
                    vertex_q.push(dest_vid);
                }
            }
    }
#ifdef SIM
    SIM_END(true);
#endif
    t2 = timer::get_usec();
    cout<<"== traversal time: "<<t2-t1<<" sec\n";
}
struct arg_t
{
    uint64_t * vertexlist; 
    uint64_t * edgelist; 
    uint16_t * vproplist;
    uint16_t * eproplist;
    uint64_t vertex_cnt;
    uint64_t edge_cnt;
    uint16_t * update;
    bool * locks;

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
    uint16_t * eproplist = arg->eproplist;
    uint16_t * update = arg->update;
    bool * locks = arg->locks;

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
            uint16_t curr_dist = vproplist[vid];
            uint32_t edge_start = vertexlist[vid];
            uint32_t edge_end = vertexlist[vid+1];

            for (unsigned j=edge_start; j<edge_end; j++)
            {
                uint64_t dest_vid = edgelist[j];
                uint16_t new_dist = curr_dist + eproplist[j];

#ifdef HMC
                if (HMC_CAS_greater_16B(&(update[dest_vid]),new_dist) > new_dist) 
                {
                    global_output_tasks[vertex_distributor(dest_vid,threadnum)+tid*threadnum].push_back(dest_vid);
                }
#else
                bool active=false;
                // spinning lock for critical section
                //  can be replaced as an atomicMin operation
                while(__sync_lock_test_and_set(&(locks[dest_vid]),1));
                if (update[dest_vid]>new_dist) 
                {
                    active = true;
                    update[dest_vid] = new_dist;
                }
                __sync_lock_release(&(locks[dest_vid]));
                
                if (active)
                {
                    global_output_tasks[vertex_distributor(dest_vid,threadnum)+tid*threadnum].push_back(dest_vid);
                }
#endif
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
        for (unsigned i=0;i<input_tasks.size();i++) 
        {
            vproplist[input_tasks[i]] = update[input_tasks[i]];
        }
        pthread_barrier_wait (&barrier);
    }

#ifdef SIM
    SIM_END(true);
#endif    
    
    if (tid!=0) pthread_exit((void*) t);

    return NULL;         
}


void parallel_SSSP(
        uint64_t * vertexlist, 
        uint64_t * edgelist, 
        uint16_t * vproplist,
        uint16_t * eproplist,
        uint64_t vertex_cnt, 
        uint64_t edge_cnt,
        uint64_t root, 
        unsigned threadnum)
{
    double t1, t2;
    
    t1 = timer::get_usec();
    
    // initializzation
    for (unsigned i=0; i<vertex_cnt; i++)
    {
        vproplist[i] = MY_INFINITY;
    }
    vproplist[root] = 0;
    
    vector<uint16_t> update(vertex_cnt, MY_INFINITY);
    update[root] = 0;

    bool * locks = new bool[vertex_cnt];
    memset(locks, 0, sizeof(bool)*vertex_cnt); 

    vector<vector<uint64_t> > global_input_tasks(threadnum);
    global_input_tasks[vertex_distributor(root,threadnum)].push_back(root);
    
    vector<vector<uint64_t> > global_output_tasks(threadnum*threadnum);
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

        args[t].update = &(update[0]);
        args[t].eproplist = eproplist;
        args[t].locks = locks;

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
        
        while(!stop)
        {
            #pragma omp barrier
            // process local queue
            stop = true;
            
            for (unsigned i=0;i<input_tasks.size();i++)
            {
                uint64_t vid=input_tasks[i];
                uint16_t curr_dist = vproplist[vid];
                uint32_t edge_start = vertexlist[vid];
                uint32_t edge_end = vertexlist[vid+1];

                for (unsigned j=edge_start; j<edge_end; j++)
                {
                    uint64_t dest_vid = edgelist[j];
                    uint16_t new_dist = curr_dist + eproplist[j];

#ifdef HMC
                    if (HMC_CAS_greater_16B(&(update[dest_vid]),new_dist) > new_dist) 
                    {
                        global_output_tasks[vertex_distributor(dest_vid,threadnum)+tid*threadnum].push_back(dest_vid);
                    }
#else
                    bool active=false;
                    // spinning lock for critical section
                    //  can be replaced as an atomicMin operation
                    while(__sync_lock_test_and_set(&(locks[dest_vid]),1));
                    if (update[dest_vid]>new_dist) 
                    {
                        active = true;
                        update[dest_vid] = new_dist;
                    }
                    __sync_lock_release(&(locks[dest_vid]));
                    
                    if (active)
                    {
                        global_output_tasks[vertex_distributor(dest_vid,threadnum)+tid*threadnum].push_back(dest_vid);
                    }
#endif
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
            for (unsigned i=0;i<input_tasks.size();i++) 
            {
                vproplist[input_tasks[i]] = update[input_tasks[i]];
            }
            #pragma omp barrier
        }
    }
#endif
    t2 = timer::get_usec();
    cout<<"== traversal time: "<<t2-t1<<" sec\n";

    delete[] locks;
}
