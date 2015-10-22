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


int16_t get_intersect_cnt(uint64_t* setA, unsigned sizeA, 
        uint64_t* setB, unsigned sizeB)
{
    int16_t ret=0;
    uint64_t* iter1=setA, *iter2=setB;

    while (iter1<(setA+sizeA) && iter2<(setB+sizeB)) 
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

unsigned seq_triangle_count(
        uint64_t * vertexlist, 
        uint64_t * edgelist, int16_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt)
{
    uint16_t ret=0;

#ifdef SIM
    SIM_BEGIN(true);
#endif
    // run triangle count now
    for (uint64_t vid=0;vid<vertex_cnt;vid++)
    {
        uint32_t edge_start = vertexlist[vid];
        uint32_t edge_end = vertexlist[vid+1];

        for (unsigned j=edge_start; j<edge_end; j++)
        {
            uint64_t dest_vid = edgelist[j];
            if (vid > dest_vid) continue;
            int16_t cnt = get_intersect_cnt(&(edgelist[vertexlist[vid]]),vertexlist[vid+1]-vertexlist[vid],
                    &(edgelist[vertexlist[dest_vid]]),vertexlist[dest_vid+1]-vertexlist[dest_vid]);
            
            vproplist[vid]+=cnt;
            vproplist[dest_vid]+=cnt;
        }
    }
    for (uint64_t vid=0;vid<vertex_cnt;vid++)
    {
        vproplist[vid] /= 2;
        ret += vproplist[vid];
    }

    ret /= 3;
#ifdef SIM
    SIM_END(true);
#endif
    return (uint16_t)ret;
}

struct arg_t
{
    uint64_t * vertexlist; 
    uint64_t * edgelist; 
    int16_t * vproplist;
    uint64_t vertex_cnt;
    uint64_t edge_cnt;
    int16_t * ret;

    unsigned tid;
    uint64_t chunk;
};

void* thread_work(void * t)
{
    struct arg_t * arg = (struct arg_t *) t;
    uint64_t * vertexlist = arg->vertexlist; 
    uint64_t * edgelist = arg->edgelist; 
    int16_t * vproplist = arg->vproplist;
    unsigned tid = arg->tid;
    uint64_t chunk = arg->chunk;
    int16_t & ret = *(arg->ret);

#ifdef SIM
    pthread_barrier_wait (&barrier);
    SIM_BEGIN(true);
#endif

    unsigned start = tid*chunk;
    unsigned end = start + chunk;
    if (end > arg->vertex_cnt) end = arg->vertex_cnt;

    // run triangle count now
    for (uint64_t vid=start;vid<end;vid++)
    {
        uint32_t edge_start = vertexlist[vid];
        uint32_t edge_end = vertexlist[vid+1];

        for (unsigned j=edge_start; j<edge_end; j++)
        {
            uint64_t dest_vid = edgelist[j];
            if (vid > dest_vid) continue;
            int16_t cnt = get_intersect_cnt(&(edgelist[vertexlist[vid]]),vertexlist[vid+1]-vertexlist[vid],
                    &(edgelist[vertexlist[dest_vid]]),vertexlist[dest_vid+1]-vertexlist[dest_vid]);
#ifdef HMC
            HMC_ADD_16B(&(vproplist[vid]),cnt);
            HMC_ADD_16B(&(vproplist[dest_vid]),cnt);
#else                
            __sync_fetch_and_add(&(vproplist[vid]),cnt);
            __sync_fetch_and_add(&(vproplist[dest_vid]),cnt);
#endif
        }
    }
    
    pthread_barrier_wait (&barrier);
    
    // tune the per-vertex count
    for (uint64_t vid=start;vid<end;vid++)
    {
        vproplist[vid] /= 2;
        __sync_fetch_and_add(&ret, vproplist[vid]);
    }

#ifdef SIM
    SIM_END(true);
#endif    
    
    if (tid!=0) pthread_exit((void*) t);

    return NULL;
}

unsigned parallel_triangle_count(
        uint64_t * vertexlist, 
        uint64_t * edgelist, int16_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt,
        unsigned threadnum)
{
    int16_t ret=0;
    uint64_t chunk = (unsigned)ceil(vertex_cnt/(double)threadnum);
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

        args[t].ret = &ret;
        args[t].chunk = chunk; 
        args[t].tid = t;
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

        unsigned start = tid*chunk;
        unsigned end = start + chunk;
        if (end > vertex_cnt) end = vertex_cnt;

        // run triangle count now
        for (uint64_t vid=start;vid<end;vid++)
        {
            uint32_t edge_start = vertexlist[vid];
            uint32_t edge_end = vertexlist[vid+1];

            for (unsigned j=edge_start; j<edge_end; j++)
            {
                uint64_t dest_vid = edgelist[j];
                if (vid > dest_vid) continue;
                int16_t cnt = get_intersect_cnt(&(edgelist[vertexlist[vid]]),vertexlist[vid+1]-vertexlist[vid],
                        &(edgelist[vertexlist[dest_vid]]),vertexlist[dest_vid+1]-vertexlist[dest_vid]);
#ifdef HMC
                HMC_ADD_16B(&(vproplist[vid]),cnt);
                HMC_ADD_16B(&(vproplist[dest_vid]),cnt);
#else                
                __sync_fetch_and_add(&(vproplist[vid]),cnt);
                __sync_fetch_and_add(&(vproplist[dest_vid]),cnt);
#endif
            }
        }
        #pragma omp barrier 
        // tune the per-vertex count
        for (uint64_t vid=start;vid<end;vid++)
        {
            vproplist[vid] /= 2;
            __sync_fetch_and_add(&ret, vproplist[vid]);
        }

    }
#endif

    ret /= 3;

    return (uint16_t)ret;
}

