//=================================================================//
// CUDA SSSP kernel
// Topological-Driven: one edge per thread, warp_centric,
//      use atomicMin
// Reference: 
// Sungpack Hong, et al. Accelerating CUDA graph algorithms 
//      at maximum warp
//=================================================================//
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#include "cudaGraph.h"

// num of vertices per warp
#define CHUNK_SZ    32
#define WARP_SZ     32

__global__ void initialize(uint32_t * d_vpl, uint32_t * d_update, bool * d_mask, uint64_t num_vertex)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid < num_vertex )
    {
        d_vpl[tid] = MY_INFINITY;
        d_update[tid] = MY_INFINITY;
        d_mask[tid] = false;
    }
}

__global__
void kernel(uint32_t * vplist, 
        uint32_t * eplist, 
        uint32_t * update, 
        bool * mask, 
        cudaGraph graph) 
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t lane_id = tid % WARP_SZ;
    uint64_t warp_id = tid / WARP_SZ;
    uint64_t v1= warp_id * CHUNK_SZ;
    uint64_t chk_sz=CHUNK_SZ;

    if((v1+CHUNK_SZ)>graph.vertex_cnt)
    {
        if ( graph.vertex_cnt>v1 ) 
            chk_sz =  graph.vertex_cnt-v1;
        else 
            return;
    }
    for(uint64_t v=v1; v< chk_sz+v1; v++)
    {
        if (mask[v])
        {
            unsigned int num_nbr = graph.get_vertex_degree(v);
            unsigned int nbr_off = graph.get_firstedge_index(v);

            uint32_t cost = vplist[v];
            for(uint64_t i=lane_id; i<num_nbr; i+=WARP_SZ)
            {
                uint64_t vid = graph.get_edge_dest(i+nbr_off);
                atomicMin(&(update[vid]), cost+eplist[i+nbr_off]);
            }
            mask[v] = false;
        }
    }
}
__global__
void kernel2(uint32_t * vplist, 
        uint32_t * update, 
        bool * mask, 
        cudaGraph graph, 
        bool *changed) 
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= graph.vertex_cnt) return;

    if (vplist[tid] > update[tid])
    {
        vplist[tid] = update[tid];
        mask[tid] = true;
        *changed = true;
    }
    else
    {
        update[tid] = vplist[tid];
    }
}


void cuda_SSSP(uint64_t * vertexlist, 
        uint64_t * degreelist, 
        uint64_t * edgelist, 
        uint32_t * vproplist,
        uint32_t * eproplist,
        uint64_t vertex_cnt, 
        uint64_t edge_cnt,
        uint64_t root)
{
    uint32_t * device_vpl = 0;
    uint32_t * device_epl = 0;
    uint32_t * device_update = 0;
    bool * device_mask = 0;
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
    unsigned int num_block_chunked = (unsigned int)ceil( num_block/(double)CHUNK_SZ )*WARP_SZ;
    
    // malloc of gpu side
    cudaErrCheck( cudaMalloc((void**)&device_vpl, vertex_cnt*sizeof(uint32_t)) );
    cudaErrCheck( cudaMalloc((void**)&device_mask, vertex_cnt*sizeof(bool)) );
    cudaErrCheck( cudaMalloc((void**)&device_update, vertex_cnt*sizeof(uint32_t)) );
    cudaErrCheck( cudaMalloc((void**)&device_epl, edge_cnt*sizeof(uint32_t)) );
    cudaErrCheck( cudaMalloc((void**)&device_over, sizeof(bool)) );

    cudaEvent_t start_event, stop_event;
    cudaErrCheck( cudaEventCreate(&start_event) );
    cudaErrCheck( cudaEventCreate(&stop_event) );
    
    // initialization
    initialize<<<num_block, num_thread_per_block>>>(device_vpl, device_update, device_mask, vertex_cnt);
    
    // prepare graph struct
    //  one for host side, one for device side
    cudaGraph h_graph, d_graph;
    // here copy only the pointers
    h_graph.read(vertexlist, degreelist, edgelist, vertex_cnt, edge_cnt);

    uint32_t zeronum=0;
    bool truenum=true;
    // memcpy from host to device
    cudaEventRecord(start_event, 0);
   
    // copy graph data to device
    h_graph.cudaGraphCopy(&d_graph);
    // set root vprop
    cudaErrCheck( cudaMemcpy(&(device_vpl[root]), &zeronum, sizeof(uint32_t), 
                cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMemcpy(&(device_mask[root]), &truenum, sizeof(bool), 
                cudaMemcpyHostToDevice) );
    // copy edge prop to device
    cudaErrCheck( cudaMemcpy(device_epl, eproplist, edge_cnt*sizeof(uint32_t), 
                cudaMemcpyHostToDevice) );

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&h2d_copy_time, start_event, stop_event);

    
    // BFS traversal
    bool stop;
    cudaEventRecord(start_event, 0);
   
    int curr=0; 
    do
    {
        // Each iteration processes 
        //      one level of BFS traversal
        stop = false;
        cudaErrCheck( cudaMemcpy(device_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) );

        kernel<<<num_block_chunked, num_thread_per_block>>>(device_vpl, device_epl,
                device_update, device_mask, d_graph);
        kernel2<<<num_block, num_thread_per_block>>>(device_vpl, device_update, 
                device_mask, d_graph, device_over);


        cudaErrCheck( cudaMemcpy(&stop, device_over, sizeof(bool), cudaMemcpyDeviceToHost) );

        curr++;
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

    printf("== iteration #: %d\n", curr);
    printf("== host->device copy time: %f ms\n", h2d_copy_time);
    printf("== device->host copy time: %f ms\n", d2h_copy_time);
    printf("== kernel time: %f ms\n", kernel_time);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    // free graph struct on device side
    d_graph.cudaGraphFree();

    cudaErrCheck( cudaFree(device_vpl) );
    cudaErrCheck( cudaFree(device_epl) );
    cudaErrCheck( cudaFree(device_update) );
    cudaErrCheck( cudaFree(device_mask) );
}

