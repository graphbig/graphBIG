//=================================================================//
// CUDA kCore kernel
// Topological-Driven: one node per thread, thread_centric,
//      use atomicAdd & atomicSub instructions
//
// Reference: 
// Alvarez-Hamelin, et al. K-Core Decomposition: A Tool for the 
//      Visualization of Large Networks.
//=================================================================//
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#include "cudaGraph.h"

__global__ void initialize(uint32_t * d_graph_property, 
        bool * d_rml, bool * d_flag, cudaGraph graph)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid < graph.vertex_cnt )
    {
        d_graph_property[tid] = graph.get_vertex_degree(tid);
        d_rml[tid] = false;
        d_flag[tid] = false;
    }
}

__global__
void kernel(uint32_t * vplist, 
        bool * rmlist, 
        bool * flaglist, 
        cudaGraph graph) 
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= graph.vertex_cnt) return;

    uint64_t vid = tid;
    if (flaglist[vid])
    {
        uint64_t start, end;
        start = graph.get_firstedge_index(vid);
        end = graph.get_edge_index_end(vid);
        for (uint64_t i=start;i<end;i++)
        {
            uint64_t dest = graph.get_edge_dest(i);
            if (rmlist[dest]==false)
                atomicSub(&(vplist[dest]), 1);
        }
        flaglist[vid] = false;
    }
}
__global__
void kernel2(uint32_t * vplist, 
        bool * rmlist, 
        bool * flaglist, 
        cudaGraph graph, 
        unsigned kcore, 
        unsigned * rm_cnt, 
        bool * d_over) 
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= graph.vertex_cnt) return;

    uint64_t vid = tid;
    if (rmlist[vid]==false)
    {
        if (vplist[vid]<kcore)
        {
            rmlist[vid]=true;
            flaglist[vid]=true;
            atomicAdd(rm_cnt, 1);
            *d_over = false;
        }
    }
}
unsigned cuda_kcore(uint64_t * vertexlist,  
        uint64_t * edgelist, uint32_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt,
        unsigned kcore)
{
    unsigned remove_cnt = 0;
    uint32_t * device_vpl = 0;
    bool * device_rml = 0;
    bool * device_flag = 0;
    bool * device_over = 0;
    unsigned * device_remove_cnt = 0;

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
    cudaErrCheck( cudaMalloc((void**)&device_rml, vertex_cnt*sizeof(bool)) );
    cudaErrCheck( cudaMalloc((void**)&device_flag, vertex_cnt*sizeof(bool)) );
    cudaErrCheck( cudaMalloc((void**)&device_remove_cnt, sizeof(unsigned)) );
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
    unsigned zeronum = 0;
    bool trueflag = true;
    cudaErrCheck( cudaMemcpy(device_remove_cnt, &zeronum, sizeof(unsigned), 
                cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMemcpy(device_over, &trueflag, sizeof(bool), 
                cudaMemcpyHostToDevice) );

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&h2d_copy_time, start_event, stop_event);

    // initialization
    initialize<<<num_block, num_thread_per_block>>>(device_vpl, device_rml, device_flag, d_graph);
    
    bool stop=false;
    cudaEventRecord(start_event, 0);
  
    do
    {
        cudaErrCheck( cudaMemcpy(device_over, &trueflag, sizeof(bool), 
                cudaMemcpyHostToDevice) );
        kernel<<<num_block, num_thread_per_block>>>(device_vpl, device_rml, device_flag, d_graph);
        kernel2<<<num_block, num_thread_per_block>>>(device_vpl, device_rml, device_flag, d_graph, kcore, device_remove_cnt, device_over);

        cudaErrCheck( cudaMemcpy(&stop, device_over, sizeof(bool), 
                cudaMemcpyDeviceToHost) );
    }while(!stop);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&kernel_time, start_event, stop_event);


    cudaEventRecord(start_event, 0);
    cudaErrCheck( cudaMemcpy(&remove_cnt, device_remove_cnt, sizeof(unsigned), 
                cudaMemcpyDeviceToHost) );
    cudaErrCheck( cudaMemcpy(vproplist, device_vpl, vertex_cnt*sizeof(uint32_t), 
                cudaMemcpyDeviceToHost) );
    
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&d2h_copy_time, start_event, stop_event);

    printf("== host->device copy time: %f ms\n", h2d_copy_time);
    printf("== device->host copy time: %f ms\n", d2h_copy_time);
    printf("== kernel time: %f ms\n", kernel_time);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    // free graph struct on device side
    d_graph.cudaGraphFree();

    cudaErrCheck( cudaFree(device_vpl) );
    cudaErrCheck( cudaFree(device_rml) );
    cudaErrCheck( cudaFree(device_flag) );
    cudaErrCheck( cudaFree(device_remove_cnt) );
    cudaErrCheck( cudaFree(device_over) );

    return remove_cnt;
}

