//=================================================================//
// CUDA BFS kernel
// Topological-Driven: one node per thread, no atomic instructions
// Reference: 
//      Pawan Harish, Accelerating large graph algorithms 
//                  on the GPU using CUDA (HiPC 2007)
//=================================================================//
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#include "cudaGraph.h"

__global__ void initialize(bool * d_graph_frontier,
                        bool * d_updating_graph_frontier,
                        bool * d_graph_visited,
                        uint32_t * d_graph_property,
                        uint64_t num_vertex)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid < num_vertex )
    {
        d_graph_frontier[tid] = false;
        d_updating_graph_frontier[tid] = false;
        d_graph_visited[tid] = false;
        d_graph_property[tid] = MY_INFINITY;
    }
}


__global__ void BFS_kernel_1(
        cudaGraph d_graph,
        bool * device_graph_frontier, 
        bool * device_updating_graph_frontier, 
        bool * device_graph_visited, 
        uint32_t * device_vpl 
        )
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
   
    if ( tid<d_graph.vertex_cnt && device_graph_frontier[tid] )
    {
        device_graph_frontier[tid] = false;
        uint64_t eidx = d_graph.get_firstedge_index(tid);
        uint64_t eidx_end = d_graph.get_edge_index_end(tid);

        for (size_t i=eidx; i<eidx_end; i++)
        {
            uint64_t vid = d_graph.get_edge_dest(i);
            if (device_graph_visited[vid]==false)
            {
                device_vpl[vid] = device_vpl[tid]+1;
                device_updating_graph_frontier[vid] = true;
            }
        }
    }
}

__global__ void BFS_kernel_2(
        bool * device_graph_frontier, 
        bool * device_updating_graph_frontier, 
        bool * device_graph_visited, 
        bool * device_over, 
        uint64_t vl_sz 
        )
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid < vl_sz && device_updating_graph_frontier[tid] )
    {
        device_graph_frontier[tid] = true;
        device_graph_visited[tid] = true;
        device_updating_graph_frontier[tid] = false;
        *device_over = true;
    } 
}


void cuda_BFS(uint64_t * vertexlist, 
        uint64_t * edgelist, uint32_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt,
        uint64_t root)
{
    uint32_t * device_vpl = 0;
    bool * device_graph_frontier = 0;
    bool * device_updating_graph_frontier = 0;
    bool * device_graph_visited = 0;
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

    cudaErrCheck( cudaMalloc((void**)&device_graph_frontier, vertex_cnt*sizeof(bool)) );
    cudaErrCheck( cudaMalloc((void**)&device_updating_graph_frontier, vertex_cnt*sizeof(bool)) );
    cudaErrCheck( cudaMalloc((void**)&device_graph_visited, vertex_cnt*sizeof(bool)) );

    cudaErrCheck( cudaMalloc((void**)&device_over, sizeof(bool)) );

    cudaEvent_t start_event, stop_event;
    cudaErrCheck( cudaEventCreate(&start_event) );
    cudaErrCheck( cudaEventCreate(&stop_event) );
    
    // initialization
    initialize<<<num_block, num_thread_per_block>>>( device_graph_frontier,
                        device_updating_graph_frontier,
                        device_graph_visited,
                        device_vpl,
                        vertex_cnt);
    
    // prepare graph struct
    //  one for host side, one for device side
    cudaGraph h_graph, d_graph;
    // here copy only the pointers
    h_graph.read(vertexlist, edgelist, vertex_cnt, edge_cnt);

    bool true_flag=true;
    uint32_t zero_flag=0;
    // memcpy from host to device
    cudaEventRecord(start_event, 0);
   
    // copy graph data to device
    h_graph.cudaGraphCopy(&d_graph);

    cudaErrCheck( cudaMemcpy(&(device_graph_frontier[root]), &true_flag, sizeof(bool), 
                cudaMemcpyHostToDevice) );  // set root vertex as the first frontier
    cudaErrCheck( cudaMemcpy(&(device_graph_visited[root]), &true_flag, sizeof(bool), 
                cudaMemcpyHostToDevice) );  // set root vertex as visited
    cudaErrCheck( cudaMemcpy(&(device_vpl[root]), &zero_flag, sizeof(uint32_t), 
                cudaMemcpyHostToDevice) );  // set root vertex as visited


    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&h2d_copy_time, start_event, stop_event);

    
    // BFS traversal
    bool stop;
    cudaEventRecord(start_event, 0);
   
    int k=0; 
    do
    {
        // Each iteration processes 
        //      one level of BFS traversal
        stop = false;
        cudaErrCheck( cudaMemcpy(device_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) );

        // step 1
        BFS_kernel_1<<<num_block, num_thread_per_block>>>(d_graph, 
                device_graph_frontier, device_updating_graph_frontier, 
                device_graph_visited, device_vpl);

        // step 2
        BFS_kernel_2<<<num_block, num_thread_per_block>>>( 
                device_graph_frontier, device_updating_graph_frontier, 
                device_graph_visited,  
                device_over, vertex_cnt);

        cudaErrCheck( cudaMemcpy(&stop, device_over, sizeof(bool), cudaMemcpyDeviceToHost) );

        k++;
    }while(stop);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&kernel_time, start_event, stop_event);


    cudaEventRecord(start_event, 0);

    cudaErrCheck( cudaMemcpy(vproplist, device_vpl, vertex_cnt*sizeof(uint32_t), 
                cudaMemcpyDeviceToHost) );
    
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&d2h_copy_time, start_event, stop_event);

    printf("== iteration #: %d\n", k);
    printf("== host->device copy time: %f ms\n", h2d_copy_time);
    printf("== device->host copy time: %f ms\n", d2h_copy_time);
    printf("== kernel time: %f ms\n", kernel_time);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    // free graph struct on device side
    d_graph.cudaGraphFree();

    cudaErrCheck( cudaFree(device_vpl) );

    cudaErrCheck( cudaFree(device_graph_frontier) );
    cudaErrCheck( cudaFree(device_updating_graph_frontier) );
    cudaErrCheck( cudaFree(device_graph_visited) );
}

