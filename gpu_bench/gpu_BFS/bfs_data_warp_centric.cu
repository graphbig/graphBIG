//=================================================================//
// CUDA BFS kernel
// Data-Driven: data-driven algorithm, global worklist in memory
//      warp centric, one edge per thread, 
//      local thread aggregate tasks first before pushing to global worklist
//      need atomicAdd for maintaining the shared worklist
// Reference: 
// Rupesh Nasre, etc. Data-driven versus Topology-driven
//      Irregular Computations on GPUs
// Duane Merrill, etc. Scalable GPU Graph Traversal
//=================================================================//
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#include "cudaGraph.h"

#define WORKLIST_SIZE   16777216
#define LOCAL_SIZE      128

#define WARP_SZ     32
#define CHUNK_SZ    32

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

__global__ void initialize(uint32_t * d_graph_property, uint64_t num_vertex)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid < num_vertex )
    {
        d_graph_property[tid] = MY_INFINITY;
    }
}

__global__
void kernel(uint32_t * vplist, cudaGraph graph, 
        my_worklist inworklist, my_worklist outworklist, unsigned curr_level) 
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t lane_id = tid % WARP_SZ;
    unsigned warp_id = tid / WARP_SZ;

    unsigned task_start = warp_id * CHUNK_SZ;
    unsigned task_end = task_start + CHUNK_SZ;

    if (task_start >= inworklist.get_item_num()) return;
    if (task_end > inworklist.get_item_num()) 
        task_end = inworklist.get_item_num();
    
    uint64_t local_worklist[LOCAL_SIZE]; 
    uint32_t work_size=0;
    
    for (unsigned id=task_start; id<task_end; id++)
    {
        uint64_t v = inworklist.get_item(id);
        uint64_t edge_ptr = graph.get_firstedge_index(v);
        uint64_t num_edge = graph.get_edge_index_end(v) - edge_ptr;

        for (uint64_t i=lane_id;i<num_edge;i+=WARP_SZ)
        {
            uint64_t vid = graph.get_edge_dest(i+edge_ptr);
            if (vplist[vid]==MY_INFINITY)
            {
                vplist[vid] = curr_level + 1;
                // push to local worklist
                local_worklist[work_size] = vid;
                work_size++;
                if (work_size==LOCAL_SIZE)
                {     
                    outworklist.pushRange(local_worklist, work_size);
                    work_size = 0;
                }
            }
        }

    }

    // push local worklist to shared worklist
    outworklist.pushRange(local_worklist, work_size);
}

void cuda_BFS(uint64_t * vertexlist, 
        uint64_t * edgelist, uint32_t * vproplist,
        uint64_t vertex_cnt, uint64_t edge_cnt,
        uint64_t root)
{
    uint32_t * device_vpl = 0;
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

    // initialize the worklists for in & out
    my_worklist worklist1, worklist2;
    worklist1.init();
    worklist2.init();

    my_worklist * in_worklist = &worklist1;
    my_worklist * out_worklist = &worklist2;

    in_worklist->host_initPush(&root, 1);

    uint32_t zeronum=0;
    // memcpy from host to device
    cudaEventRecord(start_event, 0);
   
    // copy graph data to device
    h_graph.cudaGraphCopy(&d_graph);

    cudaErrCheck( cudaMemcpy(&(device_vpl[root]), &zeronum, sizeof(uint32_t), 
                cudaMemcpyHostToDevice) );

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&h2d_copy_time, start_event, stop_event);

    
    // BFS traversal
    cudaEventRecord(start_event, 0);
   
    int curr=0;
    unsigned wl_size=1; 
    while(wl_size!=0)
    {
        // Each iteration processes 
        //      one level of BFS traversal

        num_thread_per_block = (unsigned int) wl_size;
        if (num_thread_per_block > devProp.maxThreadsPerBlock)
            num_thread_per_block = devProp.maxThreadsPerBlock;
        num_block = (unsigned int)ceil( wl_size/(double)num_thread_per_block );
        unsigned num_block_chunked = (unsigned int)ceil( num_block/(double)CHUNK_SZ )*WARP_SZ;


        kernel<<<num_block_chunked, num_thread_per_block>>>(
                device_vpl, d_graph, *in_worklist, *out_worklist, curr);
        
        my_worklist * temp=in_worklist;
        in_worklist = out_worklist;
        out_worklist = temp;
        cudaErrCheck( cudaMemcpy(&wl_size, in_worklist->end, sizeof(uint32_t), cudaMemcpyDeviceToHost) );
        out_worklist->clear();
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
}

