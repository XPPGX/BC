#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#include "headers.h"

#define MAX 100000000

double computeTime(struct timespec device_start, struct timespec device_end)
{
    const int unit_per_nano    = 1000000;
    const int unit_per_second  = 1000;
    double time = (device_end.tv_sec - device_start.tv_sec)*unit_per_second + 
                    (double)(device_end.tv_nsec - device_start.tv_nsec)/unit_per_nano;
    return time;
}
/**
 * @brief APSP-Floyd Warshall
*/
void Floyd_Warshall(struct CSR* _csr){
    printf("[Start][Floyd_Warshall]\n");
    #pragma region memoryAlloc
    int** distance_matrix = (int**)malloc(sizeof(int*) * _csr->csrVSize);

    for(int i = _csr->startNodeID ; i <= _csr->endNodeID ; i ++){

        distance_matrix[i] = (int*)malloc(sizeof(int) * _csr->csrVSize);

        for(int j = _csr->startNodeID ; j <= _csr->endNodeID ; j ++){
            distance_matrix[i][j] = MAX;
        }
    }
    #pragma endregion memoryAlloc

    // for(int i = _csr->startNodeID ; i <= _csr->endNodeID ; i ++){
    //     for(int j = _csr->startNodeID ; j <= _csr->endNodeID ; j ++){
    //         if(distance_matrix[i][j] != MAX){
    //             printf("[ERROR] (%d, %d)\n", i, j);
    //             exit(1);
    //         }
    //     }
    // }
    
    #pragma region initialDistance
    for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
        // printf("n[%d] = {", nodeID);
        distance_matrix[nodeID][nodeID] = 0;
        for(int nidx = _csr->csrV[nodeID] ; nidx < _csr->csrV[nodeID + 1] ; nidx ++){
            int nID = _csr->csrE[nidx];
            // printf("%d, ", nID);
            
            distance_matrix[nodeID][nID] = 1;
        }
        // printf("}\n");
    }
    #pragma endregion initialDistance
    

    #pragma region kernel
    for(int k = _csr->startNodeID ; k <= _csr->endNodeID ; k ++){
        for(int i = _csr->startNodeID ; i <= _csr->endNodeID ; i ++){
            for(int j = _csr->startNodeID ; j <= _csr->endNodeID ; j ++){
                
                if(distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j]){
                    distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j];
                }
            }
        }
    }
    #pragma endregion kernel

    printf("[End][Floyd_Warshall]\n");

    
}


/**
 * @brief APSP-Bellman Ford
*/
void Bellman_Ford(struct CSR* _csr, struct Graph* _graph){
    printf("[Start][Bellman_Ford]\n");
    struct Edge* edges = _graph->edges;
    int* dist_arr = (int*)malloc(sizeof(int) * _csr->csrVSize);

    for(int src = _csr->startNodeID ; src <= _csr->endNodeID ; src ++){

        for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
            dist_arr[nodeID] = MAX;
        }

        dist_arr[src] = 0;
        
        for(int time = 0 ; time < _csr->totalNodeNumber - 1 ; time ++){
            for(int edgeIdx = 0 ; edgeIdx < _graph->edgeNum ; edgeIdx ++){
                
                int node1 = edges[edgeIdx].node1;
                int node2 = edges[edgeIdx].node2;
                
                if(dist_arr[node1] != MAX && dist_arr[node1] + 1 < dist_arr[node2]){
                    dist_arr[node2] = dist_arr[node1] + 1;
                }
            }
        }
    }
    printf("[End][Bellman_Ford]\n");
}

/**
 * @brief APSP-SPFA
 * which is the version of improved Bellman Ford,
 * similar with BFS
*/
void SPFA(struct CSR* _csr, struct Graph* _graph){
    printf("[Start][SPFA]\n");
    struct qQueue* Q = InitqQueue();
    qInitResize(Q, _csr->csrVSize);

    int* dist_arr = (int*)malloc(sizeof(int) * _csr->csrVSize);
    int* inQ = (int*)calloc(_csr->csrVSize, sizeof(int)); //判斷是否在Q中，是的話對應位置就是1，否則為0
    
    
    int src;
    for(int src = _csr->startNodeID ; src <= _csr->endNodeID ; src ++){
        Q->front = 0;
        Q->rear = -1;
        
        for(int i = 0 ; i < _csr->csrVSize ; i ++){
            dist_arr[i] = MAX;
        }
        memset(inQ, 0, sizeof(int) * _csr->csrVSize);

        qPushBack(Q, src);
        dist_arr[src] = 0;
        inQ[src] = 1;

        while(!qIsEmpty(Q)){
            int curID = qPopFront(Q);
            inQ[curID] = 0;

            for(int nidx = _csr->csrV[curID] ; nidx < _csr->csrV[curID + 1] ; nidx ++){
                int nid = _csr->csrE[nidx];

                if(dist_arr[nid] > dist_arr[curID] + 1){
                    dist_arr[nid] = dist_arr[curID] + 1;

                    if(!inQ[nid]){
                        qPushBack(Q, nid);
                        inQ[curID] = 1;
                    }
                }
            }
        }
    }
    printf("[End][SPFA]\n");
}

int main(int argc, char* argv[]){
    char* datasetPath = argv[1];
    printf("exeName = %s\n", argv[0]);
    printf("datasetPath = %s\n", datasetPath);
    struct Graph* graph = buildGraph(datasetPath);
    struct CSR* csr     = createCSR(graph);
    
    struct timespec d_start, d_end;
    double time1, time2;
    double Floyd_Warshall_Time = 0;
    double Bellman_Ford_Time = 0;
    double SPFA_Time = 0;
    
    
    #pragma region Release
    /*******************************************************
     *                  Floyd Warshall                     *
    ********************************************************/
    // time1 = seconds();
    // Floyd_Warshall(csr);
    // time2 = seconds();
    // Floyd_Warshall_Time = time2 - time1;
    // printf("[ExecutionTime] Floyd_WarShallTime = %f\n", Floyd_Warshall_Time);
    // FILE* fptr = fopen("Floyd_Warshall_Time208.txt", "a+");
    // fprintf(fptr, "%f\n", Floyd_Warshall_Time);
    // fclose(fptr);
    /*******************************************************
     *                  Bellman Ford                       *
    ********************************************************/
    time1 = seconds();
    Bellman_Ford(csr, graph);
    time2 = seconds();
    Bellman_Ford_Time = time2 - time1;
    printf("[ExecutionTime] Bellman_Ford_Time = %f\n", Bellman_Ford_Time);
    FILE* fptr = fopen("Bellman_Ford_Time208.txt", "a+");
    fprintf(fptr, "%f\n", Bellman_Ford_Time);
    fclose(fptr);
    /*******************************************************
     *                        SPFA                         *
    ********************************************************/
    // time1 = seconds();
    // SPFA(csr, graph);
    // time2 = seconds();
    // SPFA_Time = time2 - time1;
    // printf("[ExecutionTime] SPFA_Time = %f\n", SPFA_Time);
    // FILE* fptr = fopen("SPFA_Time208.txt", "a+");
    // fprintf(fptr, "%f\n", SPFA_Time);
    // fclose(fptr);
    #pragma region Release
}