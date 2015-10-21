#ifndef CUDA_GRAPH_H
#define CUDA_GRAPH_H

#include <cuda.h>
#include <stdint.h>
#include <stdio.h>


#define cudaErrCheck(ans)  { \
    if ((ans)!=cudaSuccess) \
    {\
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(ans), __FILE__, __LINE__);\
        exit(ans);\
    }\
}

#define cudaLastErr { \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(error), __FILE__, __LINE__);\
        exit(-1);\
    }\
} 

// my infinity
#define MY_INFINITY    0xffffff00

typedef struct cudaGraph
{
    void read(uint64_t * vertexlist, 
            uint64_t * edgelist, 
            uint64_t v_cnt, 
            uint64_t e_cnt)
    {
        vlist = vertexlist;
        elist = edgelist;
        vertex_cnt = v_cnt;
        edge_cnt = e_cnt;
    }

    void cudaGraphAlloc()
    {
        cudaErrCheck( cudaMalloc((void**)&(vlist), (1+vertex_cnt)*sizeof(uint64_t)) );
        cudaErrCheck( cudaMalloc((void**)&(elist), edge_cnt*sizeof(uint64_t)) );
    }
    void cudaGraphCopy(struct cudaGraph * other)
    {
        other->vertex_cnt = vertex_cnt;
        other->edge_cnt = edge_cnt;

        other->cudaGraphAlloc();
        cudaErrCheck( cudaMemcpy(other->vlist, vlist, (1+vertex_cnt)*sizeof(uint64_t), 
                    cudaMemcpyHostToDevice) );
        cudaErrCheck( cudaMemcpy(other->elist, elist, edge_cnt*sizeof(uint64_t), 
                    cudaMemcpyHostToDevice) );
    }

    void cudaGraphFree()
    {
        cudaErrCheck( cudaFree(vlist) );
        cudaErrCheck( cudaFree(elist) );
    }


    __device__ inline uint64_t get_vertex_degree(uint64_t vertex_id)
    {
        return vlist[vertex_id+1]-vlist[vertex_id];
    }

    __device__ inline uint64_t get_edge_index_end(uint64_t vertex_id)
    {
        return vlist[vertex_id+1];
    }

    __device__ inline uint64_t get_firstedge_index(uint64_t vertex_id)
    {
        return vlist[vertex_id];
    }
    __device__ inline uint64_t get_edge_dest(uint64_t index)
    {
        return elist[index];
    }

    __device__ inline uint64_t * get_edge_ptr(uint64_t index)
    {
        return &(elist[index]);
    }

    uint64_t *vlist, *elist;
    uint64_t vertex_cnt, edge_cnt;
}cudaGraph;


#endif

