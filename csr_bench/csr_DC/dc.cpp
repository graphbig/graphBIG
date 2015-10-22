#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>
#include <math.h>
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
void seq_degree_centr(
        uint64_t * vertexlist, 
        uint64_t * edgelist, int16_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt)
{
    double t1, t2;
    
    t1 = timer::get_usec();

    // initializzation
    for (unsigned i=0; i<vertex_cnt; i++)
    {
        vproplist[i] = 0;
    }

    t2 = timer::get_usec();
    cout<<"== initialization time: "<<t2-t1<<" sec\n";


    t1 = timer::get_usec();
#ifdef SIM
    SIM_BEGIN(true);
#endif
    for (uint64_t i=0;i<vertex_cnt;i++)
    {
        uint64_t vid=i;
        uint32_t edge_start = vertexlist[vid];
        uint32_t edge_end = vertexlist[vid+1];

        for (unsigned j=edge_start; j<edge_end; j++)
        {
            uint64_t dest_vid = edgelist[j];
#ifdef HMC
            HMC_ADD_16B(&(vproplist[dest_vid]),1);
#else                
            vproplist[dest_vid]++;
#endif            
        }
    }
#ifdef SIM
    SIM_END(true);
#endif
    t2 = timer::get_usec();
    cout<<"== process time: "<<t2-t1<<" sec\n";
}
struct arg_t
{
    uint64_t * vertexlist; 
    uint64_t * edgelist; 
    int16_t * vproplist;
    uint64_t vertex_cnt;
    uint64_t edge_cnt;

    unsigned tid;
    unsigned threadnum;
    unsigned unit;
};
void* thread_work(void * t)
{
    struct arg_t * arg = (struct arg_t *) t;
    uint64_t * vertexlist = arg->vertexlist; 
    uint64_t * edgelist = arg->edgelist; 
    int16_t * vproplist = arg->vproplist;
    unsigned tid = arg->tid;
    unsigned unit = arg->unit;

    unsigned begin = tid*unit;
    unsigned end = begin+unit;      
    if (end > arg->vertex_cnt) end = arg->vertex_cnt;
#ifdef SIM
    pthread_barrier_wait (&barrier);
    SIM_BEGIN(true);
#endif  

    for (unsigned i=begin;i<end;i++)
    {
        uint64_t vid=i;
        uint32_t edge_start = vertexlist[vid];
        uint32_t edge_end = vertexlist[vid+1];
        
        for (unsigned j=edge_start; j<edge_end; j++)
        {
            uint64_t dest_vid = edgelist[j];
#ifdef HMC
            HMC_ADD_16B(&(vproplist[dest_vid]),1);
#else                
            __sync_fetch_and_add(&(vproplist[dest_vid]),1);
#endif
        }
    }

#ifdef SIM
    SIM_END(true);
#endif    
    
    if (tid!=0) pthread_exit((void*) t);

    return NULL;
}
void parallel_degree_centr(
        uint64_t * vertexlist, 
        uint64_t * edgelist, int16_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt,
        unsigned threadnum)
{
    double t1, t2;
    
    t1 = timer::get_usec();

    // initializzation
    for (unsigned i=0; i<vertex_cnt; i++)
    {
        vproplist[i] = 0;
    }

    t2 = timer::get_usec();
    cout<<"== initialization time: "<<t2-t1<<" sec\n";

    unsigned unit = (unsigned)ceil(vertex_cnt/(double)threadnum);

    t1 = timer::get_usec();
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
    
        args[t].tid = t;
        args[t].threadnum = threadnum;
        args[t].unit = unit;
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
    #pragma omp parallel num_threads(threadnum) 
    {
        unsigned tid = omp_get_thread_num();
        unsigned begin = tid*unit;
        unsigned end = begin+unit;      
        if (end > vertex_cnt) end = vertex_cnt;

        for (unsigned i=begin;i<end;i++)
        {
            uint64_t vid=i;
            uint32_t edge_start = vertexlist[vid];
            uint32_t edge_end = vertexlist[vid+1];
            
            for (unsigned j=edge_start; j<edge_end; j++)
            {
                uint64_t dest_vid = edgelist[j];
#ifdef HMC
                HMC_ADD_16B(&(vproplist[dest_vid]),1);
#else                
                __sync_fetch_and_add(&(vproplist[dest_vid]),1);
#endif
            }
        }
    }
#endif
    t2 = timer::get_usec();
    cout<<"== process time: "<<t2-t1<<" sec\n";
}

