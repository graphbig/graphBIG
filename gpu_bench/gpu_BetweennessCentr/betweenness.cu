//=================================================================//
// CUDA Betweenness Centr kernel
// Topological-Driven: one node per thread, thread_centric,
//      no atomicAdd instructions
//
// Reference: 
// A. E. Sariyuce, et al. Betweenness Centrality on GPUs and 
//      Heterogeneous Architectures
//=================================================================//
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#include "cudaGraph.h"

__global__ void kernel_init(unsigned * d_dist, unsigned * d_sigma, 
        float * d_delta, uint64_t num_vertex, uint64_t root)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid < num_vertex )
    {
        d_dist[tid] = MY_INFINITY;
        d_sigma[tid] = 0;
        d_delta[tid] = 0;
        if (tid==root)
        {
            d_dist[tid]=0;
            d_sigma[tid]=1;
        }
    }
}

__global__ void kernel_forward_phase(cudaGraph graph, 
        unsigned * d_dist, unsigned * d_sigma, 
        bool * d_over, unsigned curr)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= graph.vertex_cnt) return;
    if (d_dist[tid] != curr) return;

    uint64_t u = tid;
    uint64_t start, end;
    start = graph.get_firstedge_index(u);
    end = graph.get_edge_index_end(u);
    for (uint64_t i=start; i<end; i++)
    {
        uint64_t w = graph.get_edge_dest(i);
        if (d_dist[w] == MY_INFINITY)
        {
            d_dist[w] = curr+1;
            *d_over = false;
        }
        if (d_dist[w] == (curr+1))
        {
            atomicAdd(&(d_sigma[w]), d_sigma[u]);
        }
    }
}

__global__ void kernel_backward_phase(cudaGraph graph,
        unsigned * d_dist, unsigned * d_sigma,
        float * d_delta, unsigned curr)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= graph.vertex_cnt) return;
    if (d_dist[tid] != (curr-1)) return;

    uint64_t u = tid;
    float sum = 0;
    uint64_t start, end;
    start = graph.get_firstedge_index(u);
    end = graph.get_edge_index_end(u);
    for (uint64_t i=start; i<end; i++)
    {
        uint64_t w = graph.get_edge_dest(i);
        if (d_dist[w] == curr)
        {
            sum += 1.0*d_sigma[u]/d_sigma[w]*(1.0+d_delta[w]);
        }
    }
    d_delta[u] += sum;
}

__global__ void kernel_backsum_phase(cudaGraph graph, 
        float * d_BC, float * d_delta, 
        unsigned * d_dist, uint64_t root)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= graph.vertex_cnt) return;
    if (tid == root) return;
    if (d_dist[tid] == MY_INFINITY) return;

    d_BC[tid] += d_delta[tid];
}

void cuda_betweenness_centr(uint64_t * vertexlist, 
        uint64_t * edgelist, float * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt)
{
    float * device_BC = 0, * device_delta = 0;
    unsigned * device_dist = 0, * device_sigma = 0;
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
    cudaErrCheck( cudaMalloc((void**)&device_BC, vertex_cnt*sizeof(float)) );
    cudaErrCheck( cudaMemset(device_BC, 0, vertex_cnt*sizeof(float)) );
    cudaErrCheck( cudaMalloc((void**)&device_delta, vertex_cnt*sizeof(float)) );
    cudaErrCheck( cudaMalloc((void**)&device_dist, vertex_cnt*sizeof(unsigned)) );
    cudaErrCheck( cudaMalloc((void**)&device_sigma, vertex_cnt*sizeof(unsigned)) );
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

    
    cudaEventRecord(start_event, 0);
  
    for (unsigned root=0;root<vertex_cnt;root++)
    {
        kernel_init<<<num_block, num_thread_per_block>>>(device_dist, device_sigma,
                device_delta, vertex_cnt, root);

        bool stop;
        unsigned curr=0; 
        do
        {
            // Each iteration processes 
            //      one level of BFS traversal
            stop = true;
            cudaErrCheck( cudaMemcpy(device_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) );

            kernel_forward_phase<<<num_block, num_thread_per_block>>>(d_graph, device_dist, device_sigma,
                    device_over, curr);

            cudaErrCheck( cudaMemcpy(&stop, device_over, sizeof(bool), cudaMemcpyDeviceToHost) );

            curr++;
        }while(!stop);

        while(curr>1)
        {
            curr--;
            kernel_backward_phase<<<num_block, num_thread_per_block>>>(d_graph, device_dist, device_sigma,
                    device_delta, curr);
        }
        kernel_backsum_phase<<<num_block, num_thread_per_block>>>(d_graph, device_BC, device_delta, 
                device_dist, root);
    }

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&kernel_time, start_event, stop_event);


    cudaEventRecord(start_event, 0);

    cudaErrCheck( cudaMemcpy(vproplist, device_BC, vertex_cnt*sizeof(uint32_t), 
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

    cudaErrCheck( cudaFree(device_BC) );
    cudaErrCheck( cudaFree(device_delta) );
    cudaErrCheck( cudaFree(device_sigma) );
    cudaErrCheck( cudaFree(device_dist) );
    cudaErrCheck( cudaFree(device_over) );

}

