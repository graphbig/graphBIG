//=================================================================//
// CUDA BFS kernel
// Topological-Driven: one node per thread, enable kernel unrolling
//      use atomicMin for distance updates
// Reference: 
//      lonestar-GPU bfs_atomic algorithm 
//=================================================================//
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#include "cudaGraph.h"

#define WORKPERTHREAD	1
#define VERTICALWORKPERTHREAD	12	// unroll level
#define BLKSIZE 1024
#define BANKSIZE	BLKSIZE

__global__ void initialize(uint32_t * d_graph_property, uint64_t num_vertex)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid < num_vertex )
    {
        d_graph_property[tid] = MY_INFINITY;
    }
}


 
__global__
void kernel(uint32_t *vplist, cudaGraph graph, unsigned unroll, bool *changed) 
{
 	unsigned nn = WORKPERTHREAD * (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned int ii;
	__shared__ unsigned changedv[VERTICALWORKPERTHREAD * BLKSIZE]; 
	unsigned iichangedv = threadIdx.x;
	unsigned anotheriichangedv = iichangedv;
	unsigned int nprocessed = 0;

	// collect the work to be performed.
	for (unsigned node = 0; node < WORKPERTHREAD; ++node, ++nn) {
		changedv[iichangedv] = nn;
		iichangedv += BANKSIZE;
	}

	// go over the worklist and keep updating it in a BFS manner.
	while (anotheriichangedv < iichangedv) 
    {
	    nn = changedv[anotheriichangedv];
		anotheriichangedv += BANKSIZE;
	    if (nn < graph.vertex_cnt) 
        {
		    unsigned src = nn;					// source node.
			uint64_t start = graph.get_firstedge_index(src);
			uint64_t end = graph.get_edge_index_end(src);
			// go over all the target nodes for the source node.
			for (ii = start; ii < end; ++ii) 
            {
				unsigned int u = src;
				unsigned int v = graph.get_edge_dest(ii);	// target node.
				unsigned wt = 1;
				uint32_t alt = vplist[u] + wt;
				if (alt < vplist[v]) 
                {
						atomicMin(&(vplist[v]), alt);
						if (++nprocessed < unroll) 
                        {
							// add work to the worklist.
							changedv[iichangedv] = v;
							iichangedv += BANKSIZE;
						}
				}
			}
	    }
	}
	if (nprocessed) *changed = true;
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

    unsigned ProcCnt = 1;
    unsigned factor = 128;
   
    unsigned unroll = VERTICALWORKPERTHREAD; // unroll parameter <=== can be changed

    // set cuda to be sharedmem friendly
	cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferShared);
    cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	ProcCnt = deviceProp.multiProcessorCount;

    factor = (vertex_cnt + BLKSIZE * ProcCnt - 1) / (BLKSIZE * ProcCnt);

    unsigned int num_block = ProcCnt * factor;

    // malloc of gpu side
    cudaErrCheck( cudaMalloc((void**)&device_vpl, vertex_cnt*sizeof(uint32_t)) );
    cudaErrCheck( cudaMalloc((void**)&device_over, sizeof(bool)) );

    cudaEvent_t start_event, stop_event;
    cudaErrCheck( cudaEventCreate(&start_event) );
    cudaErrCheck( cudaEventCreate(&stop_event) );
    
    // initialization
    initialize<<<num_block, BLKSIZE>>>(device_vpl, vertex_cnt);
    
    // prepare graph struct
    //  one for host side, one for device side
    cudaGraph h_graph, d_graph;
    // here copy only the pointers
    h_graph.read(vertexlist, edgelist, vertex_cnt, edge_cnt);

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
    bool stop;
    cudaEventRecord(start_event, 0);
   
    int k=0; 
    do
    {
        // Each iteration processes 
        //      one level of BFS traversal
        stop = false;
        cudaErrCheck( cudaMemcpy(device_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) );

        kernel<<<num_block/WORKPERTHREAD, BLKSIZE>>>(device_vpl, d_graph, unroll, device_over);

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
}

