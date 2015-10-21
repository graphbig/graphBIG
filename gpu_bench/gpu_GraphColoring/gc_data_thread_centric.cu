//=================================================================//
// CUDA Graph Coloring kernel
// Data-Driven: one node per thread, thread_centric,
//      use atomic instruction
//
// Reference: 
// A. Grosset, et al. Evaluating Graph Coloring on GPUs
//=================================================================//
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#include "cudaGraph.h"

#define SEED 123

#define WORKLIST_SIZE   8777216
#define LOCAL_SIZE      128

// a dummy worklist that you can only push or clear
typedef struct my_worklist
{
    void init(void)
    {
        cudaErrCheck( cudaMalloc((void**)&item_array, WORKLIST_SIZE*sizeof(uint64_t)) );
        cudaErrCheck( cudaMalloc((void**)&end, sizeof(uint32_t)) );
        clear();
    }

    void clear(void)
    {
        uint32_t zeronum=0;
        cudaErrCheck( cudaMemcpy(end, &zeronum, sizeof(uint32_t), 
                cudaMemcpyHostToDevice) );
    }

    void free(void)
    {
        cudaErrCheck( cudaFree(item_array) );
        cudaErrCheck( cudaFree(end) );
    }
    __device__ void pushRange(uint64_t * from_array, uint32_t num)
    {
        uint32_t old_end = atomicAdd(end, num);
        for (uint32_t i=0;i<num;i++)
        {
            item_array[i+old_end] = from_array[i];
        }
    }
    __device__ inline uint64_t get_item(unsigned index)
    {
        return item_array[index];
    }
    __device__ inline uint32_t get_item_num(void)
    {
            return (*end);
    }
    void host_initPush(uint64_t * from_array, uint32_t num)
    {
        cudaErrCheck( cudaMemcpy(end, &num, sizeof(uint32_t), 
                cudaMemcpyHostToDevice) );
        cudaErrCheck( cudaMemcpy(item_array, from_array, num*sizeof(uint64_t), 
                cudaMemcpyHostToDevice) );
    }

    uint64_t *item_array;
    uint32_t *end;
}my_worklist;


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
        my_worklist inworklist, 
        my_worklist outworklist,
        unsigned color) 
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= inworklist.get_item_num()) return;

    uint64_t vid = inworklist.get_item(tid);
    if (vplist[vid]==MY_INFINITY)
    {
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
            outworklist.pushRange(&vid, 1);
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

    // initialize the worklists for in & out
    my_worklist worklist1, worklist2;
    worklist1.init();
    worklist2.init();

    my_worklist * in_worklist = &worklist1;
    my_worklist * out_worklist = &worklist2;

    uint64_t * tmplist = new uint64_t[vertex_cnt];
    for (unsigned i=0;i<vertex_cnt;i++)
        tmplist[i] = i;
    in_worklist->host_initPush(tmplist, vertex_cnt);
    delete []tmplist;

    // memcpy from host to device
    cudaEventRecord(start_event, 0);
   
    // copy graph data to device
    h_graph.cudaGraphCopy(&d_graph);
    // gen rand data and temprarily use vproplist to store it
    srand(SEED);
    for (unsigned i=0;i<vertex_cnt;i++)
        vproplist[i] = rand();
    cudaErrCheck( cudaMemcpy(device_rand, vproplist, vertex_cnt*sizeof(uint32_t), 
                cudaMemcpyHostToDevice) );
    
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&h2d_copy_time, start_event, stop_event);

    
    // traversal
    cudaEventRecord(start_event, 0);
   
    int curr=0;  
    unsigned wl_size=vertex_cnt;
    while(wl_size!=0)
    {
        kernel<<<num_block, num_thread_per_block>>>(device_vpl, device_rand,
                d_graph, *in_worklist, *out_worklist, curr);
        
        my_worklist * temp=in_worklist;
        in_worklist = out_worklist;
        out_worklist = temp;

        cudaErrCheck( cudaMemcpy(&wl_size, in_worklist->end, sizeof(uint32_t), cudaMemcpyDeviceToHost) );
        out_worklist->clear();
        
        num_thread_per_block = (unsigned int) wl_size;
        if (num_thread_per_block > devProp.maxThreadsPerBlock)
            num_thread_per_block = devProp.maxThreadsPerBlock;
        num_block = (unsigned int)ceil( wl_size/(double)num_thread_per_block );
        
        curr++;
    }

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

    in_worklist->free();
    out_worklist->free();

    cudaErrCheck( cudaFree(device_vpl) );
    cudaErrCheck( cudaFree(device_rand) );
}

