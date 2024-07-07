#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#include "headers.h"

#define CHECK(call){                                                           \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr,"Error: %s:%d,",__FILE__,__LINE__);                 \
        fprintf(stderr,"code: %d,reason: %s\n",error,                      \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}



__global__ void traversing_kernel(int* _g_csrV, int* _g_csrE, int* _g_dist_s, int* _g_CCs, int _startNodeID, int _endNodeID, int _csrVSize, int* _g_nodeDone){
    __shared__ int level;
    __shared__ int Flag_continue;
    __shared__ int sumDist;
    
    if(blockIdx.x < _startNodeID || blockIdx.x > _endNodeID){
        return;
    }

    //debug
    // if(blockIdx.x != 1){
    //     return;
    // }

    // if(threadIdx.x == 0){
    //     printf("blockIdx.x = %d, gridDim.x = %d\n", blockIdx.x, gridDim.x);
    // }
    for(int sourceID = blockIdx.x ; sourceID <= _endNodeID ; sourceID = sourceID + gridDim.x){
        // printf("jump\n");
        if(_g_nodeDone[sourceID] == 1){
            break;
        }

        if(threadIdx.x == 0){
            level = 0;
            Flag_continue = 1;
            sumDist = 0;

            int old_nodeDone = atomicAdd(&(_g_nodeDone[sourceID]), 1);
            _g_dist_s[(blockIdx.x * _csrVSize) + sourceID] = 0;
            // printf("blockIdx.x = %d, sourceID = %d, old_nodeDone = %d\n", blockIdx.x, sourceID, old_nodeDone);
            // printf("blockIdx.x = %d, set dist[s = %d] = %d\n", blockIdx.x, sourceID, _g_dist_s[(blockIdx.x * _csrVSize) + sourceID]);
        }

        
        __syncthreads();
        

        //BFS
        while(Flag_continue == 1){
            // printf("[HI3]\n");
            if(threadIdx.x == 0){
                Flag_continue = 0;
            }

            for(int nodeID = threadIdx.x ; nodeID <= _endNodeID ; nodeID += blockDim.x){
                if(_startNodeID <= nodeID && nodeID <= _endNodeID){
                    for(int nidx = _g_csrV[nodeID] ; nidx < _g_csrV[nodeID + 1] ; nidx ++){
                        int nid = _g_csrE[nidx];

                        if(_g_dist_s[(blockIdx.x * _csrVSize) + nid] == level && _g_dist_s[(blockIdx.x * _csrVSize) + nodeID] == -1){
                            
                            _g_dist_s[(blockIdx.x * _csrVSize) + nodeID] = level + 1;
                            // printf("threadIdx.x = %d, dist[nid = %d] = %d, Found new Node %d, level = %d\n", threadIdx.x, nid, _g_dist_s[(blockIdx.x * _csrVSize) + nid], nodeID, _g_dist_s[(blockIdx.x * _csrVSize) + nodeID]);
                            Flag_continue = 1;
                            
                        }
                    }
                }
            }

            __syncthreads();
            if(threadIdx.x == 0){
                level ++;
            }
            __syncthreads();
        }

        //sum dist
        if(threadIdx.x == 0){
            for(int nodeID = _startNodeID ; nodeID <= _endNodeID ; nodeID ++){
                if(_g_dist_s[(blockIdx.x * _csrVSize) + nodeID] > 0){
                    sumDist += _g_dist_s[(blockIdx.x * _csrVSize) + nodeID];
                }
            }

            _g_CCs[sourceID] = sumDist;
        }

        //reset
        for(int i = 0 ; i < _csrVSize ; i ++){
            _g_dist_s[(blockIdx.x * _csrVSize) + i] = -1;
        }
        __syncthreads();
    }

}

void gpu_vb(struct CSR* _csr){
    int BlockNum = 128;
    
    
    int* g_csrV;
    int* g_csrE;
    int* g_CCs;
    int* g_dist_s;
    int* g_nodeDone;


    cudaMalloc((void**)&g_csrV, sizeof(int) * _csr->csrVSize);
    cudaMalloc((void**)&g_csrE, sizeof(int) * _csr->csrESize);
    cudaMalloc((void**)&g_CCs, sizeof(int) * _csr->csrVSize);
    cudaMalloc((void**)&g_dist_s, sizeof(int) * _csr->csrVSize * BlockNum);
    cudaMalloc((void**)&g_nodeDone, sizeof(int) * _csr->csrVSize);

    cudaMemcpy(g_csrV, _csr->csrV, sizeof(int) * _csr->csrVSize, cudaMemcpyHostToDevice);
    cudaMemcpy(g_csrE, _csr->csrE, sizeof(int) * _csr->csrESize, cudaMemcpyHostToDevice);
    cudaMemset(g_CCs, 0, sizeof(int) * _csr->csrVSize);
    cudaMemset(g_dist_s, -1, sizeof(int) * _csr->csrVSize * BlockNum);
    cudaMemset(g_nodeDone, 0, sizeof(int) * _csr->csrVSize);
    //kernel
    traversing_kernel<<<BlockNum, 1024>>>(g_csrV, g_csrE, g_dist_s, g_CCs, _csr->startNodeID, _csr->endNodeID, _csr->csrVSize, g_nodeDone);
    cudaDeviceSynchronize();

    int* CCs = (int*)malloc(sizeof(int) * _csr->csrVSize);
    cudaMemcpy(CCs, g_CCs, sizeof(int) * _csr->csrVSize, cudaMemcpyDeviceToHost);
}

int main(int argc, char* argv[]){
    char* datasetPath = argv[1];
    printf("exeName = %s\n", argv[0]);
    printf("datasetPath = %s\n", datasetPath);
    struct Graph* graph = buildGraph(datasetPath);
    struct CSR* csr     = createCSR(graph);

    double time1, time2;
    
    time1 = seconds();
    gpu_vb(csr);
    time2 = seconds();
    printf("[Execution Time] gpu_vp = %f\n", time2 - time1);
    // FILE* fptr = fopen("gpu_fan_Time208.txt", "a+");
    // fprintf(fptr, "%f\n", time2 - time1);
    // fclose(fptr);
}