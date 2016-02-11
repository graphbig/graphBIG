//=================================================================//
// CUDA Graph Coloring kernel
// Topological-Driven: one node per thread, thread_centric
//
// Reference: 
// A. Grosset, et al. Evaluating Graph Coloring on GPUs
//=================================================================//
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#include "cudaGraph.h"

#define SEED 123

__global__ void initialize(uint32_t * d_vpl, uint64_t num_vertex)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid < num_vertex )
    {
        d_vpl[tid] = MY_INFINITY;
    }
}

__global__
void kernel(uint32_t * vplist, 
        uint32_t * randlist, 
        cudaGraph graph,
        uint32_t color,
        bool * device_over) 
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= graph.vertex_cnt) return;
    if (vplist[tid]==MY_INFINITY)
    {
        uint64_t vid = tid;
        uint64_t start, end;
        start = graph.get_firstedge_index(vid);
        end = graph.get_edge_index_end(vid);
        
        uint32_t local_rand = randlist[vid];
        bool found_larger=false;
        for (uint64_t i=start; i<end; i++)
        {
            uint64_t dest = graph.get_edge_dest(i);
            if (vplist[dest]<color) continue;
            if ( (randlist[dest]>local_rand) ||
                (randlist[dest]==local_rand && dest<vid))
            {
                found_larger = true;
                break;
            }
        }
        if (found_larger==false)
            vplist[vid] = color;
        else
            *device_over = false;

    }
}

void cuda_graph_coloring(uint64_t * vertexlist, 
        uint64_t * edgelist, 
        uint32_t * vproplist,
        uint64_t vertex_cnt, 
        uint64_t edge_cnt)
{
    uint32_t * device_vpl = 0;
    uint32_t * device_rand = 0;
    bool * device_over = 0;

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
    cudaErrCheck( cudaMalloc((void**)&device_rand, vertex_cnt*sizeof(uint32_t)) );
    cudaErrCheck( cudaMalloc((void**)&device_over, sizeof(bool)) );

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
    // gen rand data and temprarily use vproplist to store it
    srand(SEED);
    for (unsigned i=0;i<vertex_cnt;i++)
    {
        vproplist[i] = rand();
    }
    cudaErrCheck( cudaMemcpy(device_rand, vproplist, vertex_cnt*sizeof(uint32_t), 
                cudaMemcpyHostToDevice) );

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&h2d_copy_time, start_event, stop_event);

    
    // BFS traversal
    bool stop;
    cudaEventRecord(start_event, 0);
   
    uint32_t curr=0; 
    do
    {
        // Each iteration processes 
        //      one level of BFS traversal
        stop = true;
        cudaErrCheck( cudaMemcpy(device_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) );

        kernel<<<num_block, num_thread_per_block>>>(device_vpl, device_rand, d_graph, curr, device_over);

        cudaErrCheck( cudaMemcpy(&stop, device_over, sizeof(bool), cudaMemcpyDeviceToHost) );
        
        curr++;
    }while(!stop);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&kernel_time, start_event, stop_event);


    cudaEventRecord(start_event, 0);

    cudaErrCheck( cudaMemcpy(vproplist, device_vpl, vertex_cnt*sizeof(uint32_t), 
                cudaMemcpyDeviceToHost) );
    
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&d2h_copy_time, start_event, stop_event);

    printf("== iteration #: %d\n", curr);
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
    cudaErrCheck( cudaFree(device_rand) );
}

