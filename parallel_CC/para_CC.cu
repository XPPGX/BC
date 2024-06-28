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

struct Single_values{
    //already known
    int newEndID;
    int oriEndNodeID;
    int csrVSize;
    int csrESize;

    //get the values when traversing
    int Q_front; // a variable point to the front end of the g_Q
    int Q_next_front; //just a variable point to the front end of the g_Q in next iteration
    int Q_rear; //just a variable point to the rear end of the g_Q
    int mappingCount;
};

void quicksort_nodeID_with_degree(int* _nodes, int* _nodeDegrees, int _left, int _right);
void sortEachComp_NewID_with_degree(struct CSR* _csr, int* _newNodesID_arr, int* _newNodesDegree_arr);
__global__ void g_newID_infos_AOS_to_SOA(struct newID_info* _g_newID_infos_AOS, int* _g_newID_infos_ff, int* _g_newID_infos_r, struct Single_values* _g_csr_values);
__global__ void resetData(int* _g_dist_s, int* _g_Q, unsigned int* _g_SI, unsigned int* _g_R, struct Single_values* _g_single_values, int sourceNewID);
__global__ void mappingCount(int* _g_orderedCsrV, int* _g_orderedCsrE, int* _g_mapNodeID_New_to_Old, int* _g_nodeDone, struct Single_values* _g_single_values, int sourceNewID);
__global__ void ordinaryTraverseOneLevel(int* _g_orderedCsrV, int* _g_orderedCsrE, int* _g_mapNodeID_New_to_Old, int* _g_dist_s, int* _g_Q, struct Single_values* _g_single_values, int _Q_size);
__global__ void contributeDistToEachNodes(int* _g_dist_s, int* _g_CCs, int* _g_mapNodeID_New_to_Old, int* _g_newID_infos_ff, int* _g_newID_infos_r, int* _g_comp_newCsrOffset, int* _g_newNodesCompID, int* _g_newNodesID_arr, int _comp_size, int _sourceNewID, struct Single_values* _g_Single_values);
__global__ void copySI_toDevice(int* _g_mapNodes, unsigned int* _g_SI, int _mappingCount);
/**
 * @brief
 * preprocess : sequential => D1, AP, rebuild.
 * traversing : parallel => source traversal, neighbor sharing.
*/
void preprocess_then_parallel_sharedBased_DegreeOrder(struct CSR* _csr, int* CCs){

    #pragma region Preprocess
    D1Folding(_csr);
    AP_detection(_csr);
    AP_Copy_And_Split(_csr);
    struct newID_info* newID_infos = rebuildGraph(_csr); //rebuild graph for better memory access speed
    const int oriEndNodeID = _csr->endNodeID - _csr->apCloneCount; //原本graph的endNodeID
    
    //Sort aliveNodeID with degree
    int* newNodesID_arr     = (int*)malloc(sizeof(int) * (_csr->newEndID + 1));
    int* newNodesDegree_arr = (int*)malloc(sizeof(int) * (_csr->newEndID + 1));
    sortEachComp_NewID_with_degree(_csr, newNodesID_arr, newNodesDegree_arr);

    //整理一些要傳進GPU的常數值
    struct Single_values* Single_values = (struct Single_values*)malloc(sizeof(struct Single_values));
    Single_values->csrVSize     = _csr->csrVSize;
    Single_values->csrESize     = _csr->csrESize;
    Single_values->newEndID     = _csr->newEndID;
    Single_values->oriEndNodeID = oriEndNodeID;

    int* mapNodes       = (int*)malloc(sizeof(int) * 32);
    unsigned int* SI    = (unsigned int*)calloc(sizeof(unsigned int), (_csr->csrVSize) * 2);
    unsigned int* R     = (unsigned int*)calloc(sizeof(unsigned int), (_csr->csrVSize) * 2);

    #pragma endregion Preprocess

    #pragma region copyDataToGPU
    //declare
    int* g_CCs;

    int* g_mapNodeID_New_to_Old;
    int* g_mapNodeID_Old_to_New;
    int* g_orderedCsrV;
    int* g_orderedCsrE;
    int* g_comp_newCsrOffset; 
    int* g_newNodesCompID;
    int* g_newNodesID_arr;
    struct Single_values* g_Single_values;
    
    int* g_newID_infos_ff;
    int* g_newID_infos_r;
    struct newID_info* g_newID_infos_AOS;
    
    int* g_nodeDone;
    int* g_dist_s;
    int* g_dist_w;
    int* g_Q;
    int* g_mapNodes;
    unsigned int* g_SI; //SI, sharedBitIndex
    unsigned int* g_R; //R, relation
    
    //malloc
    cudaMalloc((void**)&g_CCs, sizeof(int) * (_csr->csrVSize));
    cudaMalloc((void**)&g_mapNodeID_New_to_Old, sizeof(int) * (_csr->csrVSize) * 2);
    cudaMalloc((void**)&g_mapNodeID_Old_to_New, sizeof(int) * (_csr->csrVSize) * 2);
    cudaMalloc((void**)&g_orderedCsrV, sizeof(int) * (_csr->csrVSize) * 2);
    cudaMalloc((void**)&g_orderedCsrE, sizeof(int) * (_csr->csrESize) * 4);
    cudaMalloc((void**)&g_comp_newCsrOffset, sizeof(int) * _csr->aliveNodeCount);
    cudaMalloc((void**)&g_newNodesCompID, sizeof(int) * _csr->csrVSize * 2);
    cudaMalloc((void**)&g_newNodesID_arr, sizeof(int) * (_csr->newEndID + 1));
    cudaMalloc((void**)&g_Single_values, sizeof(struct Single_values));

    cudaMalloc((void**)&g_newID_infos_AOS, sizeof(struct newID_info) * (_csr->newEndID + 1));
    cudaMalloc((void**)&g_newID_infos_ff, sizeof(int) * (_csr->newEndID + 1));
    cudaMalloc((void**)&g_newID_infos_r, sizeof(int) * (_csr->newEndID + 1));

    cudaMalloc((void**)&g_nodeDone, sizeof(int) * (_csr->csrVSize) * 2);
    cudaMalloc((void**)&g_dist_s, sizeof(int) * (_csr->csrVSize) * 2);
    cudaMalloc((void**)&g_dist_w, sizeof(int) * (_csr->csrVSize) * 2);
    cudaMalloc((void**)&g_Q, sizeof(int) * (_csr->csrVSize) * 2);
    cudaMalloc((void**)&g_mapNodes, sizeof(int) * 32);
    cudaMalloc((void**)&g_SI, sizeof(unsigned int) * (_csr->csrVSize) * 2);
    cudaMalloc((void**)&g_R, sizeof(unsigned int) * (_csr->csrVSize) * 2);
    //memory copy from host to GPU
    cudaMemset(g_CCs, 0, sizeof(int) * (_csr->csrVSize)); //因為有些node是AP且已在sequential取得CC了，他們的g_CCs，在kernel計算完之後會是0。
    cudaMemcpy(g_mapNodeID_New_to_Old, _csr->mapNodeID_New_to_Old, sizeof(int) * (_csr->csrVSize) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(g_mapNodeID_Old_to_New, _csr->mapNodeID_Old_to_new, sizeof(int) * (_csr->csrVSize) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(g_orderedCsrV, _csr->orderedCsrV, sizeof(int) * (_csr->csrVSize) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(g_orderedCsrE, _csr->orderedCsrE, sizeof(int) * (_csr->csrESize) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(g_comp_newCsrOffset, _csr->comp_newCsrOffset, sizeof(int) * (_csr->aliveNodeCount), cudaMemcpyHostToDevice);
    cudaMemcpy(g_newNodesCompID, _csr->newNodesCompID, sizeof(int) * (_csr->csrVSize) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(g_newNodesID_arr, newNodesID_arr, sizeof(int) * (_csr->newEndID + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(g_Single_values, Single_values, sizeof(struct Single_values), cudaMemcpyHostToDevice);

    //用一個 kenrel 去對 g_newID_infos_ff, g_newID_infos_r 賦值。
    cudaMemcpy(g_newID_infos_AOS, newID_infos, sizeof(struct newID_info) * (_csr->newEndID + 1), cudaMemcpyHostToDevice);
    g_newID_infos_AOS_to_SOA<<<(_csr->newEndID + 32 - 1 / 32), 32>>>(g_newID_infos_AOS, g_newID_infos_ff, g_newID_infos_r, g_Single_values);
    
    cudaMemset(g_nodeDone, 0, sizeof(int) * (_csr->csrVSize) * 2);
    cudaMemset(g_dist_s, -1, sizeof(int) * (_csr->csrVSize) * 2);
    cudaMemset(g_dist_w, -1, sizeof(int) * (_csr->csrVSize) * 2);
    cudaMemset(g_Q, 0, sizeof(int) * (_csr->csrVSize) * 2);
    cudaMemset(g_mapNodes, 0, sizeof(int) * 32);
    cudaMemset(g_SI, 0, sizeof(unsigned int) * (_csr->csrVSize) * 2);
    cudaMemset(g_R, 0, sizeof(unsigned int) * (_csr->csrVSize) * 2);
    
    // int* check_ff_arr = (int*)malloc(sizeof(int) * (_csr->newEndID + 1));
    // int* check_r_arr = (int*)malloc(sizeof(int) * (_csr->newEndID + 1));
    // cudaMemcpy(check_ff_arr, g_newID_infos_ff, sizeof(int) * (_csr->newEndID + 1), cudaMemcpyDeviceToHost);
    // cudaMemcpy(check_r_arr, g_newID_infos_r, sizeof(int) * (_csr->newEndID + 1), cudaMemcpyDeviceToHost);
    // for(int i = 0 ; i <= _csr->newEndID ; i ++){
    //     if(check_ff_arr[i] != newID_infos[i].ff){printf("node %d.ff wrong\n", i);}
    //     if(check_r_arr[i] != newID_infos[i].w){printf("node %d.r wrong\n", i);}
    // }
    #pragma endregion copyDataToGPU

    #pragma region Traverse
    int* nodeDone = (int*)calloc(sizeof(int*), (_csr->csrVSize) * 2);
    for(int compID = 0 ; compID <= _csr->compEndID ; compID ++){
        // printf("comp %d nodes :\n", _csr->comp_newCsrOffset[compID + 1] - _csr->comp_newCsrOffset[compID]);
        // for(int newID_idx = _csr->comp_newCsrOffset[compID + 1] - 1 ; newID_idx >= _csr->comp_newCsrOffset[compID] ; newID_idx --){
        //     int newID = newNodesID_arr[newID_idx];
        //     printf("n[%d] = {", newID);
        //     for(int nID_idx = _csr->orderedCsrV[newID] ; nID_idx < _csr->orderedCsrV[newID + 1] ; nID_idx ++){
        //         int nID = _csr->orderedCsrE[nID_idx];
        //         printf("%d, ", nID);
        //     }
        //     printf("}\n");
        // }
        // printf("\n");


        for(int newID_idx = _csr->comp_newCsrOffset[compID + 1] - 1 ; newID_idx >= _csr->comp_newCsrOffset[compID] ; newID_idx --){
            int sourceNewID = newNodesID_arr[newID_idx];
            int sourceOldID = _csr->mapNodeID_New_to_Old[sourceNewID];
            
            /**
             * 不做：
             * 1. 已經 nodeDone = 1 的 node
             * 2. CloneAP (藉由 (sourceOldID > oriEndNodeID)判斷一個node是不是 CloneAP) 
            */
            if(nodeDone[sourceNewID] == 1 || (sourceOldID > oriEndNodeID)){
                continue;
            }
            int compSize = _csr->comp_newCsrOffset[compID + 1] - _csr->comp_newCsrOffset[compID];

            nodeDone[sourceNewID] = 1;

            Single_values->Q_front = 0;
            Single_values->Q_next_front = 1;
            Single_values->Q_rear = 1;
            
            resetData<<<((_csr->csrVSize * 2 + 1023)) / 1024, 1024>>>(g_dist_s, g_Q, g_SI, g_R, g_Single_values, sourceNewID);
            int sourceNewID_degree = _csr->orderedCsrV[sourceNewID + 1] - _csr->orderedCsrV[sourceNewID];
            mappingCount<<<(sourceNewID_degree + 95) / 96, 96>>>(g_orderedCsrV, g_orderedCsrE, g_mapNodeID_New_to_Old, g_nodeDone, g_Single_values, sourceNewID);
            int mappingCount = 0;
            cudaMemcpy(&mappingCount, &(g_Single_values->mappingCount), sizeof(int), cudaMemcpyDeviceToHost);
            printf("source.degree = %d, mappingCount = %d\n", sourceNewID_degree, mappingCount);
            cudaDeviceSynchronize();

            mappingCount = 0 ; //[test] if(mappingCount < 3)
            int Q_size = 1; //means the queue size is 1 at the beginning
            int threadNum = 96; //可以用thread decide 1

            if(mappingCount < 3){ //
                

                while(Q_size > 0){
                    int blockNum = Q_size;
                    printf("[HI][1]\n");
                    cudaMemcpy(g_Single_values, Single_values, sizeof(struct Single_values), cudaMemcpyHostToDevice);
                    printf("[HI][2]\n");
                    ordinaryTraverseOneLevel<<<blockNum, threadNum>>>(g_orderedCsrV, g_orderedCsrE, g_mapNodeID_New_to_Old, g_dist_s, g_Q, g_Single_values, Q_size);
                    printf("[HI][3]\n");
                    cudaMemcpy(Single_values, g_Single_values, sizeof(struct Single_values), cudaMemcpyDeviceToHost);
                    printf("[HI][4]\n");
                    Single_values->Q_front = Single_values->Q_next_front;
                    Single_values->Q_next_front = Single_values->Q_rear;

                    Q_size = Single_values->Q_rear - Single_values->Q_front;
                    printf("h_Q_front = %d, h_Q_next_front = %d, h_Q_rear = %d, Q_size = %d\n", Single_values->Q_front, Single_values->Q_next_front, Single_values->Q_rear, Q_size);
                }

                printf("[HI][5]\n");
                printf("compSize = %d\n", compSize);
                contributeDistToEachNodes<<<(compSize + 1023)/1024, 1024>>>(g_dist_s, g_CCs, g_mapNodeID_New_to_Old, g_newID_infos_ff, g_newID_infos_r, g_comp_newCsrOffset, g_newNodesCompID, g_newNodesID_arr, compSize, sourceNewID, g_Single_values);
                cudaDeviceSynchronize();
                printf("[HI][6]\n");
            }
            else{
                mappingCount = 0;
                register int new_nID = -1;
                register int old_nID = -1;
                for(int new_nidx = _csr->orderedCsrV[sourceNewID] ; new_nidx < _csr->orderedCsrV[sourceNewID + 1] ; new_nidx ++){
                    new_nID = _csr->orderedCsrE[new_nidx];
                    old_nID = _csr->mapNodeID_New_to_Old[new_nID];
                    if(nodeDone[new_nID] == 0 && (old_nID <= oriEndNodeID)){
                        nodeDone[new_nID] = 1;
                        SI[new_nID] = 1 << mappingCount;
                        mapNodes[mappingCount] = new_nID;
                        
                        printf("\tshared new_nID %d, old_nID %d, SI = %x\n", new_nID, old_nID, SI[new_nID]);

                        mappingCount ++;
                        if(mappingCount == 32){
                            break;
                        }
                    }
                }
                
                cudaMemcpy(g_mapNodes, mapNodes, sizeof(int) * 32, cudaMemcpyHostToDevice); //Host 的 mapNodes 只有前mappingCount個有 nodes, 之後的 cell 都沒有nodes，剛好reset g_mapNodes
                copySI_toDevice<<<1, 32>>>(g_mapNodes, g_SI, mappingCount);
                // cudaMemcpyAsync(g_SI, SI, sizeof(unsigned int) * (_csr->csrVSize * 2), cudaMemcpyHostToDevice); //因為Host的SI只有那些mapNodes有值，其他SI都是0，剛好reset了g_SI。
                // cudaDeviceSynchronize();

                #pragma region SourceTraverse
                while(Q_size > 0){
                    cudaMemcpy(g_Single_values, Single_values, sizeof(struct Single_values), cudaMemcpyDeviceToHost);

                }
                #pragma endregion SourceTraverse
            }
            
            exit(1);
        }   
    }
    #pragma endregion Traverse


}

int main(int argc, char* argv[]){
    char* datasetPath = argv[1];
    printf("exeName = %s\n", argv[0]);
    printf("datasetPath = %s\n", datasetPath);
    struct Graph* graph = buildGraph(datasetPath);
    struct CSR* csr     = createCSR(graph);
    int* CCs = (int*)calloc(sizeof(int), csr->csrVSize);

    preprocess_then_parallel_sharedBased_DegreeOrder(csr, CCs);


}

void quicksort_nodeID_with_degree(int* _nodes, int* _nodeDegrees, int _left, int _right){
    if(_left > _right){
        return;
    }
    int smallerAgent = _left;
    int smallerAgentNode = -1;
    int equalAgent = _left;
    int equalAgentNode = -1;
    int largerAgent = _right;
    int largerAgentNode = -1;

    int pivotNode = _nodes[_right];
    // printf("pivot : degree[%d] = %d .... \n", pivotNode, _nodeDegrees[pivotNode]);
    int tempNode = 0;
    while(equalAgent <= largerAgent){
        #ifdef DEBUG
        // printf("\tsmallerAgent = %d, equalAgent = %d, largerAgent = %d\n", smallerAgent, equalAgent, largerAgent);
        #endif

        smallerAgentNode = _nodes[smallerAgent];
        equalAgentNode = _nodes[equalAgent];
        largerAgentNode = _nodes[largerAgent];
        
        #ifdef DEBUG
        // printf("\tDegree_s[%d] = %d, Degree_e[%d] = %d, Degree_l[%d] = %d\n", smallerAgentNode, _nodeDegrees[smallerAgentNode], equalAgentNode, _nodeDegrees[equalAgentNode], largerAgentNode, _nodeDegrees[largerAgentNode]);
        #endif

        if(_nodeDegrees[equalAgentNode] < _nodeDegrees[pivotNode]){ //equalAgentNode的degree < pivotNode的degree
            // swap smallerAgentNode and equalAgentNode
            tempNode = _nodes[smallerAgent];
            _nodes[smallerAgent] = _nodes[equalAgent];
            _nodes[equalAgent] = tempNode;

            smallerAgent ++;
            equalAgent ++;
        }
        else if(_nodeDegrees[equalAgentNode] > _nodeDegrees[pivotNode]){ //equalAgentNode的degree > pivotNode的degree
            // swap largerAgentNode and equalAgentNode
            tempNode = _nodes[largerAgent];
            _nodes[largerAgent] = _nodes[equalAgent];
            _nodes[equalAgent] = tempNode;

            largerAgent --;
        }
        else{ //equalAgentNode的degree == pivotNode的degree
            equalAgent ++;
        }

    }
    
    // exit(1);
    #ifdef DEBUG
        
    #endif

    // smallerAgent現在是pivot key的開頭
    // largerAgent現在是pivotKey的結尾
    quicksort_nodeID_with_degree(_nodes, _nodeDegrees, _left, smallerAgent - 1);
    quicksort_nodeID_with_degree(_nodes, _nodeDegrees, largerAgent + 1, _right);
}

void sortEachComp_NewID_with_degree(struct CSR* _csr, int* _newNodesID_arr, int* _newNodesDegree_arr){
    /**
     * 1. assign newID to _newNodesID_arr
     * 2. assign degree according to oldID of newID to _newNodesDegree_arr
    */
    for(int newID = 0 ; newID <= _csr->newEndID ; newID ++){
        _newNodesID_arr[newID]       = newID;
        _newNodesDegree_arr[newID]   = _csr->orderedCsrV[newID + 1] - _csr->orderedCsrV[newID];
        
        // printf("newID %d, oldID %d, degree %d\n", _newNodesID_arr[newID], _csr->mapNodeID_New_to_Old[newID], _newNodesDegree_arr[newID]);
    }

    /**
     * 在每個 component內 依照degree進行排序
    */
    for(int compID = 0 ; compID <= _csr->compEndID ; compID ++){
        int compSize = _csr->comp_newCsrOffset[compID + 1] - _csr->comp_newCsrOffset[compID];
        // printf("compID %d, compSize %d\n", compID, compSize);
        quicksort_nodeID_with_degree(_newNodesID_arr, _newNodesDegree_arr, _csr->comp_newCsrOffset[compID], _csr->comp_newCsrOffset[compID + 1] - 1);
    }

    // for(int newID_idx = 0 ; newID_idx <= _csr->newEndID ; newID_idx ++){
    //     int newID = _newNodesID_arr[newID_idx];
    //     int degree = _csr->orderedCsrV[newID + 1] - _csr->orderedCsrV[newID];
    //     printf("newID %d, degree %d, compID %d\n", newID, degree, _csr->newNodesCompID[newID]);
    // }
}


__global__ void g_newID_infos_AOS_to_SOA(struct newID_info* _g_newID_infos_AOS, int* _g_newID_infos_ff, int* _g_newID_infos_r, struct Single_values* _g_csr_values){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < (_g_csr_values->newEndID + 1)){
        
        _g_newID_infos_ff[tid]  = __ldg(&(_g_newID_infos_AOS[tid].ff));
        _g_newID_infos_r[tid]   = __ldg(&(_g_newID_infos_AOS[tid].w));
    }
}

__global__ void resetData(int* _g_dist_s, int* _g_Q, unsigned int* _g_SI, unsigned int* _g_R, struct Single_values* _g_single_values, int sourceNewID){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < (_g_single_values->csrVSize * 2)){
        _g_dist_s[tid]  = -1;
        _g_SI[tid]      = 0;
        _g_R[tid]       = 0;
    }
    if(tid == 0){
        _g_dist_s[sourceNewID]              = 0;
        _g_single_values->Q_front           = 0;
        _g_single_values->Q_next_front      = 1;
        _g_single_values->Q_rear            = 1; 
        printf("g_Q_front = %d, g_Q_rear = %d, ", _g_single_values->Q_next_front, _g_single_values->Q_rear);
        _g_Q[0] = sourceNewID; //enqueue
        printf("_g_Q[0] = %d, _g_dist_s[%d] = %d\n", _g_Q[0], sourceNewID, _g_dist_s[sourceNewID]);
        _g_single_values->mappingCount  = 0;
    }
}

__global__ void mappingCount(int* _g_orderedCsrV, int* _g_orderedCsrE, int* _g_mapNodeID_New_to_Old, int* _g_nodeDone, struct Single_values* _g_single_values, int sourceNewID){
    const int tid           = threadIdx.x + blockIdx.x * blockDim.x;
    register int new_nidx   = tid + __ldg(&(_g_orderedCsrV[sourceNewID]));
    if(new_nidx < _g_orderedCsrV[sourceNewID + 1]){
        register int new_nID = _g_orderedCsrE[new_nidx];
        register int old_nID = _g_mapNodeID_New_to_Old[new_nID];
        if((_g_nodeDone[new_nID] == 0) && (old_nID <= _g_single_values->oriEndNodeID)){
            int old_mappingCount = atomicAdd(&(_g_single_values->mappingCount), 1);
        }
    }
    if(tid == 0){
        _g_nodeDone[sourceNewID] = 1;
    }
}

__global__ void ordinaryTraverseOneLevel(int* _g_orderedCsrV, int* _g_orderedCsrE, int* _g_mapNodeID_New_to_Old, int* _g_dist_s, int* _g_Q, struct Single_values* _g_single_values, int _Q_size){
    register const int blockID = blockIdx.x; 
    // printf("bID = %d\n", blockID);
    if(blockID >= _Q_size){
        return;
    }
    
    register const int block_CurNode_Idx    = blockID + _g_single_values->Q_front;
    register const int block_CurNode_NewID  = __ldg(&(_g_Q[block_CurNode_Idx])); //整個block都在訪問CurNode_NewID的neighbors
    register const int degree               = _g_orderedCsrV[block_CurNode_NewID + 1] - _g_orderedCsrV[block_CurNode_NewID];
    //如果一個node有180個neighbor, blockDim.x = 96的話, neighborOffset = 2 代表這個block要用96個thread做兩輪,才可以把neighbor都訪問完
    register const int neighborOffset = (int)ceil(degree/(blockDim.x * 1.0)); 
    

    for(int i = 0 ; i < neighborOffset ; i ++){
        register const int thread_neighbor_idx      = _g_orderedCsrV[block_CurNode_NewID] + threadIdx.x + i * blockDim.x;
        register const int thread_neighbor_NewID    = __ldg(&(_g_orderedCsrE[thread_neighbor_idx]));
        // printf("blockID %d, threadID %d, block_CurNodeIdx = %d, block_CurNode_NewID = %d, thread_neighbor_idx = %d, thread_neighbor_NewID = %d\n", blockID, threadIdx.x, block_CurNode_Idx, block_CurNode_NewID, thread_neighbor_idx, thread_neighbor_NewID);
        if(thread_neighbor_idx < _g_orderedCsrV[block_CurNode_NewID + 1] && (atomicCAS(&(_g_dist_s[thread_neighbor_NewID]), -1, _g_dist_s[block_CurNode_NewID] + 1) == -1)){
            
            int enQ_position    = atomicAdd(&(_g_single_values->Q_rear), 1);
            _g_Q[enQ_position]  = thread_neighbor_NewID;
            printf("[EnQ] block_CurNode_NewID = %d, Q_pos = %d, \tthread_neighbor_newID = %d, \tdist = %d\n", block_CurNode_NewID, enQ_position, thread_neighbor_NewID, atomicAdd(&(_g_dist_s[thread_neighbor_NewID]), 0));
        }
    }
}

__global__ void contributeDistToEachNodes(int* _g_dist_s, int* _g_CCs, int* _g_mapNodeID_New_to_Old, int* _g_newID_infos_ff, int* _g_newID_infos_r, int* _g_comp_newCsrOffset, int* _g_newNodesCompID, int* _g_newNodesID_arr, int _comp_size, int _sourceNewID, struct Single_values* _g_Single_values){
    register const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // int laneID = threadIdx.x % 32;

    __shared__ int sharedData[2]; //sharedData[0] 是 sourceOldID, sharedData[1] 是 sourceOldID 的 compID
    if(tid == 0){
        sharedData[0] = __ldg(&(_g_mapNodeID_New_to_Old[_sourceNewID]));
        sharedData[1] = __ldg(&(_g_newNodesCompID[_sourceNewID]));;
    }
    __syncthreads();

    // if(laneID == 0){ //為了不要用太多的 __ldg
    //     sourceOldID = __ldg(&(_g_mapNodeID_New_to_Old[_sourceNewID])); // sourceNewID 的 OldID，用於在_g_CCs賦值
    //     compID      = __ldg(&(_g_newNodesCompID[_sourceNewID])); // sourceNewID 在的 compID
    // }
    // register int sharedSourceOldID =  __shfl_sync(0xffffffff, sourceOldID, 0);
    // register int sharedCompID = __shfl_sync(0xffffffff, compID, 0);

    // register const int SourceOldID = __ldg(&(_g_mapNodeID_New_to_Old[_sourceNewID])); // sourceNewID 的 OldID，用於在_g_CCs賦值
    // int compID = __ldg(&(_g_newNodesCompID[_sourceNewID]));  // sourceNewID 在的 compID
    // __syncthreads();

    if(tid < _comp_size){
        int compOtherNodeNew_idx = tid + __ldg(&(_g_comp_newCsrOffset[sharedData[1]]));
        register const int compOtherNodeNewID = __ldg(&(_g_newNodesID_arr[compOtherNodeNew_idx]));
        register const int compOtherNodeOldID = __ldg(&(_g_mapNodeID_New_to_Old[compOtherNodeNewID]));
        
        if(compOtherNodeOldID > _g_Single_values->oriEndNodeID){ //代表這個compOtherNodeOldID 是 apclone
            return;
        }
        _g_CCs[compOtherNodeOldID] += _g_newID_infos_r[_sourceNewID] * _g_dist_s[compOtherNodeNewID] + _g_newID_infos_ff[_sourceNewID];
        printf("tid = %d,\t_g_CCs[old = %d, new = %d] = %d,\t g_dist_s[new = %d] = %d\n", tid, compOtherNodeOldID, compOtherNodeNewID, _g_CCs[compOtherNodeOldID], compOtherNodeNewID, _g_dist_s[compOtherNodeNewID]);
    }
}

__global__ void copySI_toDevice(int* _g_mapNodes, unsigned int* _g_SI, int _mappingCount){
    register const int tid = threadIdx.x; //這個kernel只用一個block，該block只有32個threads.
    if(tid < _mappingCount){
        
        register const int thread_mapped_newID = __ldg(&(_g_mapNodes[tid]));
        register const int SI_value = 1 << tid;
        
        _g_SI[thread_mapped_newID] = SI_value;
        printf("tid = %d, _g_mapNodes[%d] = %d, SI_value = %x, g_SI[%d] = %x\n", tid, tid, _g_mapNodes[tid], SI_value, thread_mapped_newID, _g_SI[thread_mapped_newID]);
    }
}

__global__ void First_sharedTraverseOneLevel(int* _g_orderedCsrV, int* _g_orderedCsrE, int* _g_dist_s, int* _g_Q, struct Single_values* _g_single_values, int _Q_size, int* _g_SI, int* _g_R){
    register const int blockID = blockIdx.x;

    if(blockID >= _Q_size){
        return;
    }

    register const int block_CurNode_Idx    = blockID + _g_single_values->Q_front;
    register const int block_CurNode_NewID  = __ldg(&(_g_Q[block_CurNode_Idx]));
    register const int degree               = _g_orderedCsrV[block_CurNode_NewID + 1] - _g_orderedCsrV[block_CurNode_NewID];

    register const int neighborOffset = (int)ceil(degree/(blockDim.x * 1.0));

    for(int i = 0 ; i < neighborOffset ; i ++){
        register const int thread_neighbor_idx      = _g_orderedCsrV[block_CurNode_NewID] + threadIdx.x + i * blockDim.x;
        register const int thread_neighbor_NewID    = __ldg(&(_g_orderedCsrE[thread_neighbor_idx]));
        if(thread_neighbor_idx < _g_orderedCsrV[block_CurNode_NewID + 1]){
            
        }
    }
}