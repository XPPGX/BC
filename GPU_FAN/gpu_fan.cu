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

__global__ void init_dist(int* _g_dist_s, int _startNodeID, int _endNodeID, int _sourceID){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(_startNodeID <= tid && tid <= _endNodeID){
        if(tid == _sourceID){
            _g_dist_s[tid] = 0;
        }
        else{
            _g_dist_s[tid] = -1;
            
        }
        // printf("g_dist[%d] = %d\n", tid, _g_dist_s[tid]);
    }
}

__global__ void apsp_kernel(int* _g_edgeList1, int* _g_edgeList2, int* _g_dist_s, int _edgeNum, int _level, int* _g_continue_flag){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < _edgeNum){
        int sNode = _g_edgeList1[tid];
        int eNode = _g_edgeList2[tid];
        // printf("tid = %d, (%d, %d), _g_dist_s[%d] = %d\n", tid, sNode, eNode, sNode, _g_dist_s[sNode]);
        
        if(_g_dist_s[sNode] == _level){
            if(_g_dist_s[eNode] == -1){
                *_g_continue_flag = 1;
                _g_dist_s[eNode] = _level + 1;
                // printf("\ttid %d, find newID %d, _g_dist_s[%d] = %d\n", tid, eNode, eNode, _g_dist_s[eNode]);
            }
        }
    }
}

__global__ void sumDist(int* _g_CCs, int* _g_dist_s, int _startNodeID, int _endNodeID){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if( _startNodeID <= tid && tid <= _endNodeID){
        _g_CCs[tid] += _g_dist_s[tid];
    }
}

void gpu_fan(struct Graph* _graph, struct CSR* _csr){
    
    int* edgeList1 = (int*)malloc(sizeof(int) * _csr->csrESize);
    int* edgeList2 = (int*)malloc(sizeof(int) * _csr->csrESize);
    int count = 0;
    for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
        for(int nidx = _csr->csrV[nodeID] ; nidx < _csr->csrV[nodeID + 1] ; nidx ++){
            int nid = _csr->csrE[nidx];
            edgeList1[count] = nodeID;
            edgeList2[count] = nid;
            // printf("(%d, %d), e_idx = %d\n", edgeList1[count], edgeList2[count], count);
            count ++;
        }
    }

    // for(int e_idx = 0 ; e_idx < _graph->edgeNum ; e_idx ++){
    //     edgeList1[e_idx] = _graph->edges[e_idx].node1;
    //     edgeList2[e_idx] = _graph->edges[e_idx].node2;
    //     printf("(%d, %d), e_idx = %d\n", edgeList1[e_idx], edgeList2[e_idx], e_idx);
    // }
    
    int level;
    int continue_flag; 

    int* g_continue_flag;
    int* g_edgeList1;
    int* g_edgeList2;
    int* g_dist_s;
    int* g_CCs;

    cudaMalloc((void**)&g_continue_flag, sizeof(int));
    cudaMalloc((void**)&g_edgeList1, sizeof(int) * _csr->csrESize);
    cudaMalloc((void**)&g_edgeList2, sizeof(int) * _csr->csrESize);
    cudaMalloc((void**)&g_dist_s, sizeof(int) * _csr->csrVSize);
    cudaMalloc((void**)&g_CCs, sizeof(int) * _csr->csrVSize);

    cudaMemset(g_continue_flag, 1, sizeof(int));
    cudaMemset(g_CCs, 0, sizeof(int) * _csr->csrVSize);
    cudaMemcpy(g_edgeList1, edgeList1, sizeof(int) * _csr->csrESize, cudaMemcpyHostToDevice);
    cudaMemcpy(g_edgeList2, edgeList2, sizeof(int) * _csr->csrESize, cudaMemcpyHostToDevice);
    
    
    for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
        continue_flag = 1;
        level = 0;
        // printf("\t\tsource = %d\n", nodeID);
        // printf("\t\t[Init]\n");
        init_dist<<<((_csr->csrESize + 31) / 32), 32>>>(g_dist_s, _csr->startNodeID, _csr->endNodeID, nodeID);
        cudaDeviceSynchronize();
        // printf("\t\t[Init] done\n");
        while(continue_flag == 1){
            // printf("\n");
            // printf("\tcontinue_flag = %d, now_level = %d\n", continue_flag, level);
            // printf("\n");

            cudaMemset(g_continue_flag, 0, sizeof(int));
            apsp_kernel<<<(_csr->csrESize + 31) / 32, 32>>>(g_edgeList2, g_edgeList1, g_dist_s, _csr->csrESize, level, g_continue_flag);
            cudaDeviceSynchronize();
            cudaMemcpy(&continue_flag, g_continue_flag, sizeof(int), cudaMemcpyDeviceToHost);

            level ++;
            
        }
        sumDist<<<(_csr->csrVSize + 31) / 32, 32>>>(g_CCs, g_dist_s, _csr->startNodeID, _csr->endNodeID);
    }
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
    gpu_fan(graph, csr);
    time2 = seconds();
    printf("[Execution Time] gpu_fan = %f\n", time2 - time1);
    FILE* fptr = fopen("gpu_fan_Time208.txt", "a+");
    fprintf(fptr, "%f\n", time2 - time1);
    fclose(fptr);
}