//=================================================================//
// CUDA Connected Component kernel
// Topological-Driven: one edge per thread, no atomic instructions

// Reference:
// Jyothish Soman, et. al, SOME GPU ALGORITHMS FOR GRAPH CONNECTED 
//      COMPONENTS AND SPANNING TREE
//=================================================================//
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#include "cudaGraph.h"

#define CHUNK_SZ 2

__global__ void initialize(uint64_t * d_parents, uint64_t * d_shadow, bool * d_mask, uint64_t * d_edge_src, cudaGraph graph)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= graph.vertex_cnt) return;

    d_parents[tid] = tid;
    d_shadow[tid] = tid;

    uint64_t start, end;
    start = graph.get_firstedge_index(tid);
    end = graph.get_edge_index_end(tid);
    for (uint64_t i=start; i<end; i++)
    {
        d_mask[i] = false;
        d_edge_src[i] = tid;
    }

}

// checking all edges in parallel
__global__ void kernel_hooking(
        uint64_t * d_parents, 
        uint64_t * d_shadow, 
        bool * d_mask, 
        uint64_t * d_edge_src, 
        bool * d_over,
        unsigned iter,
        cudaGraph graph)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t chunk_id = tid * CHUNK_SZ;

    if (chunk_id >= graph.edge_cnt) return;
   
    uint64_t end_id = chunk_id + CHUNK_SZ;
    if (end_id > graph.edge_cnt) 
       end_id = graph.edge_cnt; 
    for (uint64_t eid=chunk_id;eid<end_id;eid++)
    {
        if (d_mask[eid]) continue;

        uint64_t src = d_edge_src[eid];
        uint64_t dest = graph.get_edge_dest(eid);

        if (d_parents[src] != d_parents[dest])
        {
            uint64_t min, max;
            //uint64_t mn, mx;
            if (d_parents[src]>d_parents[dest])
            {
                max = d_parents[src];
                min = d_parents[dest];
                //mx = src;
                //mn = dest;
            }
            else
            {
                max = d_parents[dest];
                min = d_parents[src];
                //mx = dest;
                //mn = src;
            }

            if ((iter%2)==0) // even iterations
            {
                d_shadow[min] = max;
                //d_parents[mn] = max;
            }
            else // odd iterations
            {
                d_shadow[max] = min;
                //d_parents[mx] = min;
            }
            *d_over = false;
        }
        else
        {
            d_mask[eid] = true;
        }
    }
}

__global__ void kernel_update(
        uint64_t * d_parents, 
        uint64_t * d_shadow, 
        unsigned vertex_cnt) 
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= vertex_cnt) return;
    d_parents[tid] = d_shadow[tid];
}

__global__ void kernel_pointer_jumping(
        uint64_t * d_parents, 
        uint64_t * d_shadow, 
        unsigned vertex_cnt)
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= vertex_cnt) return;

    uint64_t parent = d_parents[tid];
    while(d_parents[parent]!=parent)
    {
        parent = d_parents[parent];
    }

    d_shadow[tid] = parent; 
}

void cuda_connected_comp(uint64_t * vertexlist, 
        uint64_t * edgelist, uint64_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt)
{
    uint64_t * device_parents = 0;
    uint64_t * device_shadow = 0;
    bool * device_mask = 0;
    uint64_t * device_edge_src = 0;
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

    unsigned int num_thread_per_block2 = (unsigned int) ceil(edge_cnt/(double)CHUNK_SZ);
    if (num_thread_per_block2 > devProp.maxThreadsPerBlock)
        num_thread_per_block2 = devProp.maxThreadsPerBlock;
    unsigned int num_block2 = (unsigned int)ceil( edge_cnt/(double)num_thread_per_block2/(double)CHUNK_SZ );


    // malloc of gpu side
    cudaErrCheck( cudaMalloc((void**)&device_parents, vertex_cnt*sizeof(uint64_t)) );
    cudaErrCheck( cudaMalloc((void**)&device_shadow, vertex_cnt*sizeof(uint64_t)) );
    cudaErrCheck( cudaMalloc((void**)&device_mask, edge_cnt*sizeof(bool)) );
    cudaErrCheck( cudaMalloc((void**)&device_edge_src, edge_cnt*sizeof(uint64_t)) );
    cudaErrCheck( cudaMalloc((void**)&device_over, sizeof(bool)) );

    cudaEvent_t start_event, stop_event;
    cudaErrCheck( cudaEventCreate(&start_event) );
    cudaErrCheck( cudaEventCreate(&stop_event) );
    
        
    // prepare graph struct
    //  one for host side, one for device side
    cudaGraph h_graph, d_graph;
    // here copy only the pointers
    h_graph.read(vertexlist, edgelist, vertex_cnt, edge_cnt);

    // memcpy from host to device
    cudaEventRecord(start_event, 0);
   
    // copy graph data to device
    h_graph.cudaGraphCopy(&d_graph);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&h2d_copy_time, start_event, stop_event);

    // initialization
    initialize<<<num_block, num_thread_per_block>>>(device_parents, device_shadow, device_mask, device_edge_src, d_graph);

    cudaEventRecord(start_event, 0);
 
    bool stop = false;
    unsigned iter=0;
    do
    {
        stop = true;
        cudaErrCheck( cudaMemcpy(device_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) );

        // merging sub trees
        kernel_hooking<<<num_block2, num_thread_per_block2>>>(device_parents, device_shadow, device_mask, 
                device_edge_src, device_over, iter, d_graph);
        
        kernel_update<<<num_block, num_thread_per_block>>>(device_parents, device_shadow, vertex_cnt);
        //cudaDeviceSynchronize();
        cudaErrCheck( cudaMemcpy(&stop, device_over, sizeof(bool), cudaMemcpyDeviceToHost) );
        iter++;

        // multiple level pointer jumping
        kernel_pointer_jumping<<<num_block, num_thread_per_block>>>(device_parents, device_shadow, vertex_cnt);
        kernel_update<<<num_block, num_thread_per_block>>>(device_parents, device_shadow, vertex_cnt);
        //cudaDeviceSynchronize();
    }while(!stop);

    
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&kernel_time, start_event, stop_event);


    cudaEventRecord(start_event, 0);

    cudaErrCheck( cudaMemcpy(vproplist, device_parents, vertex_cnt*sizeof(uint64_t), 
                cudaMemcpyDeviceToHost) );
    
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&d2h_copy_time, start_event, stop_event);
#ifndef ENABLE_VERIFY
    printf("== host->device copy time: %f ms\n", h2d_copy_time);
    printf("== device->host copy time: %f ms\n", d2h_copy_time);
    printf("== kernel time: %f ms\n", kernel_time);
    fflush(stdout);
#endif
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    // free graph struct on device side
    d_graph.cudaGraphFree();

    cudaErrCheck( cudaFree(device_parents) );
    cudaErrCheck( cudaFree(device_shadow) );
    cudaErrCheck( cudaFree(device_mask) );
    cudaErrCheck( cudaFree(device_edge_src) );
    cudaErrCheck( cudaFree(device_over) );
}

