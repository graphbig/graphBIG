//=================================================================//
// CUDA BFS kernel
// Topological-Driven: one node per thread, thread_centric,
//      use atomicAdd instruction
//
// Reference: 
// T. Schank. Algorithmic Aspects of Triangle-Based Network Analysis
//=================================================================//
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#include "cudaGraph.h"

#define STACK_SZ 2048
#define SEED 123

__device__ void quicksort(uint64_t array[], unsigned len)
{
    uint64_t left = 0, stack[STACK_SZ], pos = 0, seed = SEED;
    for ( ; ; )                                             /* outer loop */
    {
        for (; left+1 < len; len++)                  /* sort left to len-1 */
        {
            if (pos == STACK_SZ) len = stack[pos = 0];  /* stack overflow, reset */
            uint64_t pivot = array[left+seed%(len-left)];  /* pick random pivot */
            seed = seed*69069+1;                /* next pseudorandom number */
            stack[pos++] = len;                    /* sort right part later */
            for (unsigned right = left-1; ; )   /* inner loop: partitioning */
            {
                while (array[++right] < pivot);  /* look for greater element */
                while (pivot < array[--len]);    /* look for smaller element */
                if (right >= len) break;           /* partition point found? */
                uint64_t temp = array[right];
                array[right] = array[len];                  /* the only swap */
                array[len] = temp;
            }                            /* partitioned, continue left part */
        }
        if (pos == 0) break;                               /* stack empty? */
        left = len;                             /* left to right is sorted */
        len = stack[--pos];                      /* get next range to sort */
    }
}
__device__ unsigned get_intersect_cnt(uint64_t* setA, unsigned sizeA, uint64_t* setB, unsigned sizeB)
{
    unsigned ret=0;
    unsigned iter1=0, iter2=0;
    while (iter1<sizeA && iter2<sizeB) 
    {
        if (setA[iter1] < setB[iter2]) 
            iter1++;
        else if (setA[iter1] > setB[iter2]) 
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

__global__ void initialize(uint32_t * d_graph_property, uint64_t num_vertex)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid < num_vertex )
    {
        d_graph_property[tid] = 0;
    }
}

// sort the outgoing edges accoring to their ID order
__global__
void kernel_step1(uint32_t * vplist, cudaGraph graph) 
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= graph.vertex_cnt) return;

    uint64_t start;
    start = graph.get_firstedge_index(tid);
    quicksort(graph.get_edge_ptr(start), graph.get_vertex_degree(tid));
}

// get neighbour set intersections of the vertices with edge links
__global__
void kernel_step2(uint32_t * vplist, cudaGraph graph) 
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= graph.vertex_cnt) return;

    uint64_t start, end;
    start = graph.get_firstedge_index(tid);
    end = start + graph.get_vertex_degree(tid);
    for (uint64_t i=start; i<end; i++)
    {
        uint64_t dest = graph.get_edge_dest(i);
        if (tid > dest) continue; // skip reverse edges
        uint64_t dest_start = graph.get_firstedge_index(dest);
        unsigned cnt = get_intersect_cnt(graph.get_edge_ptr(start),
                graph.get_vertex_degree(tid), graph.get_edge_ptr(dest_start),
                graph.get_vertex_degree(dest));

        atomicAdd(&(vplist[tid]), cnt);
        atomicAdd(&(vplist[dest]), cnt);
    }

}

// reduction to get the total count
__global__
void kernel_step3(uint32_t * vplist, cudaGraph graph, unsigned * d_tcount) 
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= graph.vertex_cnt) return;

    vplist[tid] = vplist[tid] / 2;

    atomicAdd(d_tcount, vplist[tid]);
}


unsigned cuda_triangle_count(uint64_t * vertexlist, 
        uint64_t * edgelist, uint32_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt)
{
    unsigned tcount = 0;
    uint32_t * device_vpl = 0;
    unsigned * device_tcount = 0;

    float h2d_copy_time = 0; // host to device data transfer time
    float d2h_copy_time = 0; // device to host data transfer time
    float kernel_time = 0;   // kernel execution time

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp,device);


    // Try to use as many threads as possible so that each thread
    //      is processing one vertex. If max thread is reached, 
    //      split them into multiple blocks.
    unsigned int num_thread_per_block = (unsigned int) vertex_cnt;
    if (num_thread_per_block > devProp.maxThreadsPerBlock)
        num_thread_per_block = devProp.maxThreadsPerBlock;
    unsigned int num_block = (unsigned int)ceil( vertex_cnt/(double)num_thread_per_block );

    // malloc of gpu side
    cudaErrCheck( cudaMalloc((void**)&device_vpl, vertex_cnt*sizeof(uint32_t)) );
    cudaErrCheck( cudaMalloc((void**)&device_tcount, sizeof(unsigned)) );

    cudaEvent_t start_event, stop_event;
    cudaErrCheck( cudaEventCreate(&start_event) );
    cudaErrCheck( cudaEventCreate(&stop_event) );
    
    // initialization
    initialize<<<num_block, num_thread_per_block>>>(device_vpl, vertex_cnt);
    
    // prepare graph struct
    //  one for host side, one for device side
    cudaGraph h_graph, d_graph;
    // here copy only the pointers
    h_graph.read(vertexlist, edgelist, vertex_cnt, edge_cnt);

    // memcpy from host to device
    cudaEventRecord(start_event, 0);
   
    // copy graph data to device
    h_graph.cudaGraphCopy(&d_graph);
    unsigned zeronum = 0;
    cudaErrCheck( cudaMemcpy(device_tcount, &zeronum, sizeof(unsigned), 
                cudaMemcpyHostToDevice) );
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&h2d_copy_time, start_event, stop_event);

    
    cudaEventRecord(start_event, 0);
  
    kernel_step1<<<num_block, num_thread_per_block>>>(device_vpl, d_graph);
    kernel_step2<<<num_block, num_thread_per_block>>>(device_vpl, d_graph);
    kernel_step3<<<num_block, num_thread_per_block>>>(device_vpl, d_graph, device_tcount);
    cudaErrCheck( cudaMemcpy(&tcount, device_tcount, sizeof(unsigned), cudaMemcpyDeviceToHost) );

    tcount /= 3;

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&kernel_time, start_event, stop_event);


    cudaEventRecord(start_event, 0);

    cudaErrCheck( cudaMemcpy(vproplist, device_vpl, vertex_cnt*sizeof(uint32_t), 
                cudaMemcpyDeviceToHost) );
    
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&d2h_copy_time, start_event, stop_event);
#ifndef ENABLE_VERIFY
    printf("== host->device copy time: %f ms\n", h2d_copy_time);
    printf("== device->host copy time: %f ms\n", d2h_copy_time);
    printf("== kernel time: %f ms\n", kernel_time);
#endif
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    // free graph struct on device side
    d_graph.cudaGraphFree();

    cudaErrCheck( cudaFree(device_vpl) );
    cudaErrCheck( cudaFree(device_tcount) );

    return tcount;
}

