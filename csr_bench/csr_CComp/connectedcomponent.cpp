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

unsigned global_label = 0;

inline unsigned vertex_distributor(uint64_t vid, unsigned threadnum)
{
    return vid%threadnum;
}

unsigned seq_CC(
        uint64_t * vertexlist, 
        uint64_t * edgelist, uint16_t * vproplist, uint16_t * labellist,
        uint64_t vertex_cnt, uint64_t edge_cnt)
{
    double t1, t2;
    
    t1 = timer::get_usec();

    // initializzation
    for (unsigned i=0; i<vertex_cnt; i++)
    {
        vproplist[i] = MY_INFINITY;
        labellist[i] = MY_INFINITY;
    }

    std::queue<uint64_t> vertex_queue;
    uint64_t root;    
    unsigned ret = 0;
    t2 = timer::get_usec();
    cout<<"== initialization time: "<<t2-t1<<" sec\n";

    t1 = timer::get_usec();
#ifdef SIM
        SIM_BEGIN(true);
#endif           
    
    for (uint64_t i=0;i<vertex_cnt;i++)
    {
        if (vproplist[i] != MY_INFINITY) continue;

        ret++;
        root = i;
        vertex_queue.push(root);
        unsigned label = global_label; global_label++;
       
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
                    MY_INFINITY,curr_level+1) == MY_INFINITY)
                {
                    HMC_CAS_equal_16B(&(labellist[dest_vid]),MY_INFINITY,label);
#else
                if (vproplist[dest_vid]==MY_INFINITY)
                {
                    vproplist[dest_vid] = curr_level+1;
                    labellist[dest_vid] = label;
#endif                
                    vertex_queue.push(dest_vid);
                }
            }
        }
    }
#ifdef SIM
        SIM_END(true);
#endif
    t2 = timer::get_usec();
    cout<<"== traversal time: "<<t2-t1<<" sec\n";

    return ret;
}

struct arg_t
{
    uint64_t * vertexlist; 
    uint64_t * edgelist; 
    uint16_t * vproplist;
    uint16_t * labellist;
    uint64_t vertex_cnt;
    uint64_t edge_cnt;

    vector<vector<uint64_t> > * global_input_tasks_ptr;
    vector<vector<uint64_t> > * global_output_tasks_ptr;
    unsigned tid;
    unsigned threadnum;
    bool * stop;
    uint64_t * root;
    unsigned * ret;
};

void* thread_work(void * t)
{
    struct arg_t * arg = (struct arg_t *) t;
    vector<vector<uint64_t> > & global_input_tasks = *(arg->global_input_tasks_ptr);
    vector<vector<uint64_t> > & global_output_tasks = *(arg->global_output_tasks_ptr);
    uint64_t * vertexlist = arg->vertexlist; 
    uint64_t * edgelist = arg->edgelist; 
    uint16_t * vproplist = arg->vproplist;
    uint16_t * labellist = arg->labellist;
    unsigned tid = arg->tid;
    bool & stop = *(arg->stop);
    unsigned threadnum = arg->threadnum;
    uint64_t & root = *(arg->root);
    unsigned & ret = *(arg->ret);

    vector<uint64_t> & input_tasks = global_input_tasks[tid];
#ifdef SIM
    pthread_barrier_wait (&barrier);
    SIM_BEGIN(true);
#endif       
    while(root < arg->vertex_cnt)
    {
        if (tid == 0)
        {
            vproplist[root] = 0;
            labellist[root] = global_label;
            global_input_tasks[vertex_distributor(root, threadnum)].push_back(root);
            stop = false;
        }
        pthread_barrier_wait (&barrier);

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
                    {
                        HMC_CAS_equal_16B(&(labellist[dest_vid]), MY_INFINITY, global_label);
#else
                    if (__sync_bool_compare_and_swap(&(vproplist[dest_vid]), 
                                MY_INFINITY,curr_level+1))
                    {
                        labellist[dest_vid] = global_label;
#endif
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
        if (tid == 0)
        {
            ret++;

            for (;root<arg->vertex_cnt;root++)
            {
                if (vproplist[root]==MY_INFINITY) break;
            }
            global_label++;
        }
        pthread_barrier_wait (&barrier);
    }
#ifdef SIM
    SIM_END(true);
#endif    
    
    if (tid!=0) pthread_exit((void*) t);

    return NULL;
}

unsigned parallel_CC(
        uint64_t * vertexlist, 
        uint64_t * edgelist, uint16_t * vproplist,
        uint16_t * labellist,
        uint64_t vertex_cnt, uint64_t edge_cnt,
        unsigned threadnum)
{
    double t1, t2;
    
    t1 = timer::get_usec();

    // initializzation
    for (unsigned i=0; i<vertex_cnt; i++)
    {
        vproplist[i] = MY_INFINITY;
        labellist[i] = MY_INFINITY;
    }
    uint64_t root = 0;
    unsigned ret = 0;

    vector<vector<uint64_t> > global_input_tasks(threadnum);
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
        args[t].labellist = labellist;
        args[t].vertex_cnt = vertex_cnt;
        args[t].edge_cnt = edge_cnt;

        args[t].global_input_tasks_ptr = &global_input_tasks;
        args[t].global_output_tasks_ptr = &global_output_tasks;
    
        args[t].tid = t;
        args[t].stop = &stop;
        args[t].root = &root;
        args[t].ret= &ret;
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
        
        while(root < vertex_cnt)
        {
            if (tid == 0)
            {
                vproplist[root] = 0;
                labellist[root] = root;
                global_input_tasks[vertex_distributor(root, threadnum)].push_back(root);
            }


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
            if (tid == 0)
            {
                ret++;

                for (;root<vertex_cnt;root++)
                {
                    if (vproplist[root]==MY_INFINITY) break;
                }
                stop = false;
            }
            pthread_barrier_wait (&barrier);
        }
    }
#endif
    t2 = timer::get_usec();
    cout<<"== traversal time: "<<t2-t1<<" sec\n";

    return ret;
}

