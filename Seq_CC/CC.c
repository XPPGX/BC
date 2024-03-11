#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#include "headers.h"

#pragma region DefineLabel
// #define DEBUG
// #define CheckDistAns
#pragma endregion //DefineLabel

#pragma region globalVar
int tempSourceID = 0;
#pragma endregion //globalVar

inline void resetQueue(struct qQueue* _Q){
    _Q->front = 0;
    _Q->rear = -1;
    //Q->size如果不變，就不須memcpy
}



int* computeCC(struct CSR* _csr, int* _CCs){
    // showCSR(_csr);
    int* dist_arr = (int*)calloc(sizeof(int), _csr->csrVSize);

    struct qQueue* Q = InitqQueue();
    qInitResize(Q, _csr->csrVSize);

    int sourceID;

    #ifdef CheckDistAns
    sourceID = tempSourceID;
    #else
    sourceID = _csr->startNodeID;
    #endif

    for(; sourceID <= _csr->endNodeID ; sourceID ++){
        memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);
        resetQueue(Q);

        qPushBack(Q, sourceID);
        dist_arr[sourceID]  = 0;
        // printf("\nSourceID = %2d ...\n", sourceID);       
        int currentNodeID   = -1;
        int neighborNodeID  = -1;


        while(!qIsEmpty(Q)){
            currentNodeID = qPopFront(Q);
            
            // printf("%2d ===\n", currentNodeID);

            for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
                neighborNodeID = _csr->csrE[neighborIndex];
                // printf("\t%2d meet %2d, dist_arr[%2d] = %2d\n", currentNodeID, neighborNodeID, neighborNodeID, dist_arr[neighborNodeID]);
                if(dist_arr[neighborNodeID] == -1){
                    qPushBack(Q, neighborNodeID);
                    dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;

                    // printf("\tpush %2d to Q, dist_arr[%2d] = %2d\n", neighborNodeID, neighborNodeID, dist_arr[neighborNodeID]);
                }
            }
        }

        for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
            _CCs[nodeID] = _CCs[nodeID] + dist_arr[nodeID];
            // printf("CC[%2d] = %2d\n", nodeID, _CCs[nodeID]);
        }
        #ifdef CheckDistAns
        break;
        #endif
    }

    free(Q->dataArr);
    free(Q);

    #ifndef CheckDistAns
    free(dist_arr);
    #endif

    return dist_arr;
}

/**
 * @brief the SI bit can serve 32 bit only
*/
void computeCC_shareBased(struct CSR* _csr, int* _CCs){
    // showCSR(_csr);
    
    int* dist_arr = (int*)malloc(sizeof(int) * _csr->csrVSize);
    int* neighbor_dist_ans = (int*)malloc(sizeof(int) * _csr->csrVSize);

    struct qQueue* Q = InitqQueue();
    qInitResize(Q, _csr->csrVSize);

    //record that nodes which haven't been source yet
    int* nodeDone = (int*)calloc(sizeof(int), _csr->csrVSize);

    //record nodes belongs to which neighbor of source
    int* mapping_SI                 = (int*)malloc(sizeof(int) * 32);
    unsigned int* sharedBitIndex    = (unsigned int*)calloc(sizeof(unsigned int), _csr->csrVSize); //for recording blue edge bitIndex
    unsigned int* relation          = (unsigned int*)calloc(sizeof(unsigned int), _csr->csrVSize); //for recording red edge bitIndex
    
    for(int sourceID = _csr->startNodeID ; sourceID <= _csr->endNodeID ; sourceID ++){
        if(nodeDone[sourceID] == 1){
            continue;
        }
        nodeDone[sourceID] = 1;

        // printf("SourceID = %2d\n", sourceID);

        memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);
        
        resetQueue(Q);
        
        dist_arr[sourceID] = 0;
        qPushBack(Q, sourceID);

        register int currentNodeID  = -1;
        register int neighborNodeID = -1;
        register int neighborIndex  = -1;

        //each neighbor mapping to bit_SI, if it haven'ts been source yet
        int mappingCount = 0;
        for(neighborIndex = _csr->csrV[sourceID] ; neighborIndex < _csr->csrV[sourceID + 1] ; neighborIndex ++){
            neighborNodeID = _csr->csrE[neighborIndex];

            if(nodeDone[neighborNodeID] == 0){

                sharedBitIndex[neighborNodeID] = 1 << mappingCount;
                mapping_SI[mappingCount] = neighborNodeID;

                // printf("sharedBitIndex[%6d] = %8x,\tmapping_SI[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], mappingCount, mapping_SI[mappingCount]);
                #ifdef DEBUG
                printf("sharedBitIndex[%2d] = %8x,\tmapping_SI[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], mappingCount, mapping_SI[mappingCount]);
                #endif
                
                mappingCount ++;

                //Record to 32 bit only
                if(mappingCount == 32){
                    break;
                }

            }
        }
        
        if(mappingCount < 3){
            //把sharedBitIndex重設。
            for(int mappingIndex = 0 ; mappingIndex < mappingCount ; mappingIndex ++){
                int nodeID = mapping_SI[mappingIndex];
                sharedBitIndex[nodeID] = 0;
            }
            memset(mapping_SI, 0, sizeof(int) * 32);

            #pragma region Ordinary_BFS_Forward_Traverse

            #ifdef DEBUG
            printf("\n####      Source %2d Ordinary BFS Traverse      ####\n\n", sourceID);
            #endif

            while(!qIsEmpty(Q)){
                currentNodeID = qPopFront(Q);

                #ifdef DEBUG
                printf("\tcurrentNodeID = %2d ... dist = %2d\n", currentNodeID, dist_arr[currentNodeID]);
                #endif

                for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborNodeID = _csr->csrE[neighborIndex];

                    if(dist_arr[neighborNodeID] == -1){
                        qPushBack(Q, neighborNodeID);
                        dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;

                        #ifdef DEBUG
                        printf("\t\t[1]dist[%2d] = %2d\n", neighborNodeID, dist_arr[neighborNodeID]);
                        #endif
                    }
                }
            }
            #pragma endregion //Ordinary_BFS_Forward_Traverse



            #pragma region distAccumulation_pushBased
            //Update CC in the way of pushing is better for parallelism because of the it will not need to wait atomic operation on single address,
            //it can update all value in each CC address in O(1) time.
            for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
                _CCs[nodeID] += dist_arr[nodeID];
            }
            #pragma endregion //distAccumulation_pushBased



            #pragma region checkingDistAns
            #ifdef CheckAns
            CC_CheckDistAns(_csr, _CCs, sourceID, dist_arr);
            #endif
            #pragma endregion //checkingDistAns

            

        }
        else{

            #pragma region SourceTraverse
            //main source traversal : for getting the dist of each node from source
            #ifdef DEBUG
            printf("\n####      Source %2d First traverse...      ####\n\n", sourceID);
            #endif

            while(!qIsEmpty(Q)){
                currentNodeID = qPopFront(Q);

                #ifdef DEBUG
                printf("currentNodeID = %2d ... dist = %2d\n", currentNodeID, dist_arr[currentNodeID]);
                #endif

                

                for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborNodeID = _csr->csrE[neighborIndex];

                    if(dist_arr[neighborNodeID] == -1){//traverse new succesor and record its SI
                        qPushBack(Q, neighborNodeID);
                        dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;
                        sharedBitIndex[neighborNodeID] |= sharedBitIndex[currentNodeID];

                        #ifdef DEBUG
                        printf("\t[1]unvisited_SI[%2d] => %2x, dist[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], neighborNodeID, dist_arr[neighborNodeID]);
                        #endif
                        
                        // if(sourceID == 5 && (neighborNodeID == 4 || neighborNodeID == 6)){
                        //     printf("\t[1]currentNodeID = %2d(dist %2d, SI %2x), neighborNodeID = %d(dist %2d, SI %2x)\n", currentNodeID, dist_arr[currentNodeID], sharedBitIndex[currentNodeID], neighborNodeID, dist_arr[neighborNodeID], sharedBitIndex[neighborNodeID]);
                        // }
                    }
                    else if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){ //traverse to discovered succesor and record its SI
                        sharedBitIndex[neighborNodeID] |= sharedBitIndex[currentNodeID];    
                        
                        #ifdef DEBUG
                        printf("\t[2]visited_SI[%2d] => %2x, dist[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], neighborNodeID, dist_arr[neighborNodeID]);
                        #endif

                        // if(sourceID == 5 && (neighborNodeID == 4 || neighborNodeID == 6)){
                        //     printf("\t[2]currentNodeID = %2d(dist %2d, SI %2x), neighborNodeID = %d(dist %2d, SI %2x)\n", currentNodeID, dist_arr[currentNodeID], sharedBitIndex[currentNodeID], neighborNodeID, dist_arr[neighborNodeID], sharedBitIndex[neighborNodeID]);
                        // }
                    }
                    else if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] && currentNodeID < neighborNodeID){ //traverse to discovered neighbor which is at same level as currentNodeID
                        relation[currentNodeID]     |= sharedBitIndex[neighborNodeID] & (~sharedBitIndex[currentNodeID]);
                        relation[neighborNodeID]    |= sharedBitIndex[currentNodeID] & (~sharedBitIndex[neighborNodeID]);

                        #ifdef DEBUG
                        printf("\t[3]Red edge found(%2d, %2d), ", currentNodeID, neighborNodeID);
                        printf("relation[%2d] = %2x, relation[%2d] = %2x\n", currentNodeID, relation[currentNodeID], neighborNodeID, relation[neighborNodeID]);
                        #endif

                        // if(sourceID == 5 && (neighborNodeID == 4 || neighborNodeID == 6)){
                        //     printf("\t[3]currentNodeID = %2d(dist %2d, re %2x), neighborNodeID = %d(dist %2d, re %2x)\n", currentNodeID, dist_arr[currentNodeID], relation[currentNodeID], neighborNodeID, dist_arr[neighborNodeID], relation[neighborNodeID]);
                        // }
                    }
                }
            }

            //second source traversal : for handle the red edge
            #ifdef DEBUG
            printf("\n####      Source %2d Second traverse...      ####\n\n", sourceID);
            #endif

            Q->front = 0;
            while(!qIsEmpty(Q)){
                currentNodeID = qPopFront(Q);

                #ifdef DEBUG
                printf("currentNodeID = %2d ... dist = %2d ... relation = %x\n", currentNodeID, dist_arr[currentNodeID], relation[currentNodeID]);
                #endif

                for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborNodeID = _csr->csrE[neighborIndex];

                    if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){
                        relation[neighborNodeID] |= relation[currentNodeID];
                        
                        #ifdef DEBUG
                        printf("\t[4]relation[%2d] = %2x\n", neighborNodeID, relation[neighborNodeID]);
                        #endif

                        // if(sourceID == 5 && (neighborNodeID == 4 || neighborNodeID == 6)){
                        //     printf("\t[4]currentNodeID = %2d(dist %2d, re %2x), neighborNodeID = %d(dist %2d, re %2x)\n", currentNodeID, dist_arr[currentNodeID], relation[currentNodeID], neighborNodeID, dist_arr[neighborNodeID], relation[neighborNodeID]);
                        // }
                    }
                }
            }
            #pragma endregion //SourceTraverse

            
            #pragma region sourceDistAccumulation_pushBased
            for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
                _CCs[nodeID] += dist_arr[nodeID];
            }
            #pragma endregion //distAccumulation_pushBased



            #pragma region neighborOfSource_GetDist
            //recover the data from source to neighbor of source
            for(int sourceNeighborIndex = 0 ; sourceNeighborIndex < mappingCount ; sourceNeighborIndex ++){
                memset(neighbor_dist_ans, 0, sizeof(int));

                int sourceNeighborID = mapping_SI[sourceNeighborIndex];
                unsigned int bit_SI = 1 << sourceNeighborIndex;

                nodeDone[sourceNeighborID] = 1;

                #ifdef DEBUG
                printf("\nnextBFS = %2d, bit_SI = %x\n", sourceNeighborID, bit_SI);
                #endif

                for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
                    if((sharedBitIndex[nodeID] & bit_SI) > 0){ //要括號，因為"比大小優先於邏輯運算"
                        neighbor_dist_ans[nodeID] = dist_arr[nodeID] - 1;
                        // printf("\t[5]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", nodeID, neighbor_dist_ans[nodeID], nodeID, sharedBitIndex[nodeID]);
                    }
                    else{
                        neighbor_dist_ans[nodeID] = dist_arr[nodeID] + 1;
                        // printf("\t[6]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", nodeID, neighbor_dist_ans[nodeID], nodeID, sharedBitIndex[nodeID]);
                        if((relation[nodeID] & bit_SI) > 0){
                            neighbor_dist_ans[nodeID] --;
                            // printf("\t[7]neighbor_dist_ans[%2d] = %2d, relation[%2d] = %x\n", nodeID, neighbor_dist_ans[nodeID], nodeID, relation[nodeID]);
                        }
                    }
                    
                }



                #pragma region neighborDistAccumulation_pushBased
                for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
                    _CCs[nodeID] += neighbor_dist_ans[nodeID];
                }
                #pragma endregion //neighborDistAccumulation_pushBased



                #pragma region checkingDistAns
                #ifdef CheckAns
                CC_CheckDistAns(_csr, _CCs, sourceNeighborID, neighbor_dist_ans);
                #endif
                #pragma endregion //checkingDistAns
            }
            #pragma endregion //neighborOfSource_GetDist

            //reset the SI & relation arrays
            memset(relation, 0, sizeof(unsigned int) * _csr->csrVSize);
            memset(sharedBitIndex, 0, sizeof(unsigned int) * _csr->csrVSize);
        }
    }
    // printf("\n\n[CC_sharedBased] Done!\n");
}

/**
 * @brief 
 * 1. source traverse the component, get distance from source to each node that is still in component
 * 2. update CC of source itself
 * 3. if all node in the remaining component had been source once, then go to step 4, else go to step 1
 * 4. let each d1Node to get its parent's CC to update CC of d1Node itself
*/
void compute_D1_CC(struct CSR* _csr, int* _CCs){
    
    int* dist_arr = (int*)calloc(sizeof(int), _csr->csrVSize);
    struct qQueue* Q = InitqQueue();
    qInitResize(Q, _csr->csrVSize);

    D1Folding(_csr);

    #pragma region SourceTraverse_With_ff_And_represent
    //In this block, we get the CC of each remaining node in the component
    int sourceID = -1;
    for(int notD1NodeIndex = 0 ; notD1NodeIndex < _csr->ordinaryNodeCount ; notD1NodeIndex ++){
        sourceID = _csr->notD1Node[notD1NodeIndex];
        
        #ifdef DEBUG
        printf("sourceID = %2d, ff[%2d] = %2d, represent[%2d] = %2d\n", sourceID, sourceID, _csr->ff[sourceID], sourceID, _csr->representNode[sourceID]);
        #endif

        memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);
        resetQueue(Q);
        
        qPushBack(Q, sourceID);
        dist_arr[sourceID] = 0;

        int currentNodeID   = -1;
        int neighborNodeID  = -1;

        while(!qIsEmpty(Q)){
            currentNodeID = qPopFront(Q);

            #ifdef DEBUG
            printf("currentNodeID = %2d, dist_arr[%2d] = %2d ===\n", currentNodeID, currentNodeID, dist_arr[currentNodeID]);
            #endif

            for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                neighborNodeID = _csr->csrE[neighborIndex];

                #ifdef DEBUG
                printf("\t%2d meet %2d, dist_arr[%2d] = %2d\n", currentNodeID, neighborNodeID, neighborNodeID, dist_arr[neighborNodeID]);
                #endif

                if(dist_arr[neighborNodeID] == -1){
                    qPushBack(Q, neighborNodeID);
                    dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;

                    //update CC (push-based)
                    _CCs[neighborNodeID] += _csr->ff[sourceID] + dist_arr[neighborNodeID] * _csr->representNode[sourceID];

                    #ifdef DEBUG
                    printf("\t\tpush %2d to Q, dist_arr[%2d] = %2d, _CCs[%2d] = %2d\n", neighborNodeID, neighborNodeID, dist_arr[neighborNodeID], neighborNodeID, _CCs[neighborNodeID]);
                    #endif
                }
            }
        }

        //each sourceNode update its CC with self.ff
        _CCs[sourceID] += _csr->ff[sourceID];

        // break;
    }
    #pragma endregion //SourceTraverse_With_ff_And_represent


    #pragma region d1NodeDistRecovery
    printf("_csr->totalNodeNumber = %2d\n", _csr->totalNodeNumber);
    int d1NodeID        = -1;
    int d1NodeParentID  = -1;
    for(int d1NodeIndex = _csr->degreeOneNodesQ->rear ; d1NodeIndex >= 0 ; d1NodeIndex --){
        d1NodeID = _csr->degreeOneNodesQ->dataArr[d1NodeIndex];
        d1NodeParentID = _csr->D1Parent[d1NodeID];
        _CCs[d1NodeID] = _CCs[d1NodeParentID] + _csr->totalNodeNumber - 2 * _csr->representNode[d1NodeID];
        printf("d1NodeID = %2d, _CCs[%2d] = %2d, ParentID = %2d, _CCs[%2d] = %2d\n", d1NodeID, d1NodeID, _CCs[d1NodeID], d1NodeParentID, d1NodeParentID, _CCs[d1NodeParentID]);
    }
    #pragma endregion //d1NodeDistRecovery



}

//寫個檢查CC答案的function

void CC_CheckAns(struct CSR* _csr, int* TrueCC_Ans, int* newCC_Ans){
    for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
        if(TrueCC_Ans[nodeID] != newCC_Ans[nodeID]){
            printf("[ERROR] CC ans : TrueCC_Ans[%2d] = %2d, newCC_Ans[%2d] = %2d\n", nodeID, TrueCC_Ans[nodeID], nodeID, newCC_Ans[nodeID]);
            exit(1);
        }
    }
}

int CC_CheckDistAns(struct CSR* _csr, int* _CCs, int _tempSourceID, int* dist){
    tempSourceID = _tempSourceID;
    int* ans = computeCC(_csr, _CCs);

    printf("[Ans Checking] SourceID = %2d ... ", _tempSourceID);

    for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
        if(dist[nodeID] != ans[nodeID]){
            printf("[ERROR] dist[%2d] = %2d, ans[%2d] = %2d\n", nodeID, dist[nodeID], nodeID, ans[nodeID]);
            exit(1);
        }
    }
    
    printf("Correct !!!!\n");
    free(ans);
}


int main(int argc, char* argv[]){
    char* datasetPath = argv[1];
    printf("exeName = %s\n", argv[0]);
    printf("datasetPath = %s\n", datasetPath);
    struct Graph* graph = buildGraph(datasetPath);
    struct CSR* csr     = createCSR(graph);
    
    double CC_shareBasedTime    = 0;
    double CC_ori               = 0;
    double D1_CC_ori            = 0;
    double D1_CC_shareBasedTime = 0;

    double time1                = 0;
    double time2                = 0;

    int* CCs        = (int*)calloc(sizeof(int), csr->csrVSize);
    computeCC(csr, CCs);
    for(int nodeID = csr->startNodeID ; nodeID <= csr->endNodeID ; nodeID ++){
        printf("CC[%2d] = %2d\n", nodeID, CCs[nodeID]);
    }

    // computeCC(csr, CCs);
    // time1 = seconds();
    // computeCC_shareBased(csr, CCs);
    // time2 = seconds();
    // CC_shareBasedTime = time2 - time1;

    
    // time1 = seconds();
    // computeCC(csr, CCs);
    // time2 = seconds();
    // CC_ori = time2 - time1;
    
    int* tempCCs    = (int*)calloc(sizeof(int), csr->csrVSize);
    time1 = seconds();
    compute_D1_CC(csr, tempCCs);
    time2 = seconds();
    D1_CC_ori = time2 - time1;

    CC_CheckAns(csr, CCs, tempCCs);

    
}
