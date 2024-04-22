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
// #define CheckCC_Ans
#pragma endregion //DefineLabel

#pragma region globalVar
int tempSourceID        = 0;
int CheckedNodeCount    = 0;
int* TrueCC_Ans;
#pragma endregion //globalVar

inline void resetQueue(struct qQueue* _Q){
    _Q->front   = 0;
    _Q->rear    = -1;
    //Q->size如果不變，就不須memcpy
}


#pragma region Function_computing

int* computeCC(struct CSR* _csr, int* _CCs){
    // showCSR(_csr);
    int* dist_arr       = (int*)calloc(sizeof(int), _csr->csrVSize);

    struct qQueue* Q    = InitqQueue();
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

        #ifdef DEBUG
        printf("\nSourceID = %2d ...\n", sourceID);
        #endif      

        int currentNodeID   = -1;
        int neighborNodeID  = -1;


        while(!qIsEmpty(Q)){
            currentNodeID = qPopFront(Q);
            
            #ifdef DEBUG
            printf("%2d ===\n", currentNodeID);
            #endif

            for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
                neighborNodeID = _csr->csrE[neighborIndex];

                #ifdef DEBUG
                printf("\t%2d meet %2d, dist_arr[%2d] = %2d\n", currentNodeID, neighborNodeID, neighborNodeID, dist_arr[neighborNodeID]);
                #endif

                if(dist_arr[neighborNodeID] == -1){
                    qPushBack(Q, neighborNodeID);
                    dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;

                    #ifdef DEBUG
                    printf("\tpush %2d to Q, dist_arr[%2d] = %2d\n", neighborNodeID, neighborNodeID, dist_arr[neighborNodeID]);
                    #endif
                }
            }
        }

        for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
            _CCs[nodeID] = _CCs[nodeID] + dist_arr[nodeID];

            #ifdef DEBUG
            printf("CC[%2d] = %2d\n", nodeID, _CCs[nodeID]);
            #endif
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
    
    int* dist_arr           = (int*)malloc(sizeof(int) * _csr->csrVSize);
    int* neighbor_dist_ans  = (int*)malloc(sizeof(int) * _csr->csrVSize);

    struct qQueue* Q        = InitqQueue();
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

        //each neighbor of sourceID mapping to bit_SI, if it haven't been source yet
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
            #ifdef CheckDistAns
            // CC_CheckDistAns(_csr, _CCs, sourceID, dist_arr);
            #endif

            #ifdef CheckCC_Ans
            dynamic_CC_trace_Ans(_csr, _CCs, sourceID);
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

            #ifdef CheckCC_Ans
            dynamic_CC_trace_Ans(_csr, _CCs, sourceID);
            #endif

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
                #ifdef CheckDistAns
                // CC_CheckDistAns(_csr, _CCs, sourceNeighborID, neighbor_dist_ans);
                #endif

                #ifdef CheckCC_Ans
                dynamic_CC_trace_Ans(_csr, _CCs, sourceNeighborID);
                #endif

                #pragma endregion //checkingDistAns
            }
            #pragma endregion //neighborOfSource_GetDist

            //reset the SI & relation arrays
            memset(relation, 0, sizeof(unsigned int) * _csr->csrVSize);
            memset(sharedBitIndex, 0, sizeof(unsigned int) * _csr->csrVSize);
        }
    }
    printf("\n");
    // printf("\n\n[CC_sharedBased] Done!\n");
}

/**
 * @brief
 * 1. Perform D1Folding.
 * 2. source traverse the component, get distance from source to each node that is still in component
 * 3. update CC of source itself
 * 4. if all node in the remaining component had been source once, then go to step 4, else go to step 1
 * 5. let each d1Node to get its parent's finished CC to update CC of d1Node itself
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


    #pragma region d1Node_Dist_And_CC_Recovery
    // printf("_csr->totalNodeNumber = %2d\n", _csr->totalNodeNumber);
    int d1NodeID        = -1;
    int d1NodeParentID  = -1;
    for(int d1NodeIndex = _csr->degreeOneNodesQ->rear ; d1NodeIndex >= 0 ; d1NodeIndex --){
        d1NodeID        = _csr->degreeOneNodesQ->dataArr[d1NodeIndex];
        d1NodeParentID  = _csr->D1Parent[d1NodeID];
        _CCs[d1NodeID]  = _CCs[d1NodeParentID] + _csr->totalNodeNumber - 2 * _csr->representNode[d1NodeID];
        // printf("d1NodeID = %2d, _CCs[%2d] = %2d, ParentID = %2d, _CCs[%2d] = %2d\n", d1NodeID, d1NodeID, _CCs[d1NodeID], d1NodeParentID, d1NodeParentID, _CCs[d1NodeParentID]);
    }
    #pragma endregion //d1Node_Dist_And_CC_Recovery


    // #pragma region WriteCC_To_txt
    // for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
    //     printf("%d %d\n", nodeID, _CCs[nodeID]);
    // }
    // #pragma endregion
}


/**
 * @brief
 * 1. Perform D1Folding
 * 2. Mapping each neighbor of source, if any neighbor of source haven't been source yet
*/
void compute_D1_CC_shareBased(struct CSR* _csr, int* _CCs){
    
    int* dist_arr           = (int*)malloc(sizeof(int) * _csr->csrVSize);
    int* neighbor_dist_ans  = (int*)malloc(sizeof(int) * _csr->csrVSize);
    
    struct qQueue* Q        = InitqQueue();
    qInitResize(Q, _csr->csrVSize);

    //record that nodes which haven't been source yet
    int* nodeDone = (int*)calloc(sizeof(int), _csr->csrVSize);

    //record nodes belongs to which neighbor of source
    int* mapping_SI                 = (int*)malloc(sizeof(int) * 32);
    unsigned int* sharedBitIndex    = (unsigned int*)calloc(sizeof(unsigned int), _csr->csrVSize); //SBI
    unsigned int* relation          = (unsigned int*)calloc(sizeof(unsigned int), _csr->csrVSize);
    
    //Folding D1
    D1Folding(_csr);
    
    /**
     * After D1Folding, we've got two lists which are:
     * 1. _csr->d1Node_List         : _csr->degreeOneNodesQ->dataArr
     * 2. _csr->notD1Node_List      : _csr->notD1Node 
    */

    int ordinaryTraversalCount  = 0;
    int sharingTraversalCount   = 0;
    int sourceID = -1;
    for(int notD1NodeIndex = 0 ; notD1NodeIndex < _csr->ordinaryNodeCount ; notD1NodeIndex ++){
        sourceID = _csr->notD1Node[notD1NodeIndex];
        if(nodeDone[sourceID] == 1){
            continue;
        }
        
        nodeDone[sourceID] = 1; //= =, don't forget this line

        #ifdef DEBUG
        printf("\nsourceID = %2d...\n", sourceID);
        #endif

        //reset dist_arr        
        memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);

        //reset Queue
        resetQueue(Q);

        //Init dist_arr[sourceID]
        dist_arr[sourceID] = 0;

        //push sourceID into Q
        qPushBack(Q, sourceID);

        register int currentNodeID  = -1;
        register int neighborNodeID = -1;
        register int neighborIndex  = -1;

        // Count neighbors of sourceID which are not done yet
        int mappingCount = 0;
        for(int neighborIndex = _csr->csrV[sourceID] ; neighborIndex < _csr->oriCsrV[sourceID + 1] ; neighborIndex ++){
            neighborNodeID = _csr->csrE[neighborIndex];
            if(nodeDone[neighborNodeID] == 0){
                mappingCount ++;
            }
        }

        //decide to use the sharing strategy or not
        if(mappingCount < 3){ // perform ordinary traverse
            ordinaryTraversalCount ++;
            
            while(!qIsEmpty(Q)){
                currentNodeID = qPopFront(Q);

                for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborNodeID = _csr->csrE[neighborIndex];
                    
                    if(dist_arr[neighborNodeID] == -1){
                        qPushBack(Q, neighborNodeID);
                        dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;

                        _CCs[neighborNodeID] += _csr->ff[sourceID] + dist_arr[neighborNodeID] * _csr->representNode[sourceID];
                    }
                }
            }
            _CCs[sourceID] += _csr->ff[sourceID];

        }
        else{ //perform traversal with sharing strategy
            sharingTraversalCount ++;

            #pragma region mappingNeighbor
            mappingCount = 0;
            //each neighbor of sourceID mapping to SBI, if it haven't been source yet
            for(neighborIndex = _csr->csrV[sourceID] ; neighborIndex < _csr->oriCsrV[sourceID + 1] ; neighborIndex ++){
                neighborNodeID = _csr->csrE[neighborIndex];

                if(nodeDone[neighborNodeID] == 0){
                    
                    sharedBitIndex[neighborNodeID] = 1 << mappingCount;
                    mapping_SI[mappingCount] = neighborNodeID;
                    
                    #ifdef DEBUG
                    printf("sharedBitIndex[%2d] = %8x, \tmapping_SI[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], mappingCount, mapping_SI[mappingCount]);
                    #endif

                    mappingCount ++;

                    //Record to 32 bit only
                    if(mappingCount == 32){
                        break;
                    }
                    
                }
            }
            #pragma endregion //mappingNeighbor
            


            #pragma region SourceTraverse
            /**
             * Main source traverse for getting some information:
             * 
             * ##        First Traverse        ##
             * 1. dist              : distance from source to each node in the component
             * 2. sharedBitIndex    : record each node that can be found first by which neighbor of source
             * ##################################
             * 
             * ##        Second Traverse        ##
             * 3. relation          : record each node that should remain the distance when source are some specific nodes
             * ###################################
             * 
             * @todo 
             * 1. consider another way to replace "the 3 if statement", the way of branchless technique may be a good choice
             * 2. A way : when currentNodeID find unvisited node v, then just push v into Q, and don't update its dist,
             *          until v becomes currentNodeID, then assign the distance for v.
            */

            #ifdef DEBUG
            printf("\n####      Source %2d 1st traverse...      ####\n\n", sourceID);
            #endif

            while(!qIsEmpty(Q)){
                currentNodeID = qPopFront(Q);
                
                #ifdef DEBUG
                printf("currentNodeID = %2d ... dist = %2d ... SI = %x ... relation = %x\n", currentNodeID, dist_arr[currentNodeID], sharedBitIndex[currentNodeID], relation[currentNodeID]);
                #endif

                for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborNodeID = _csr->csrE[neighborIndex];

                    if(dist_arr[neighborNodeID] == -1){ //traverse new succesor and update its SI
                        qPushBack(Q, neighborNodeID);
                        dist_arr[neighborNodeID]        = dist_arr[currentNodeID] + 1;
                        sharedBitIndex[neighborNodeID] |= sharedBitIndex[currentNodeID];
                        
                        //update CC (push-based)
                        _CCs[neighborNodeID] += _csr->ff[sourceID] + dist_arr[neighborNodeID] * _csr->representNode[sourceID];

                        #ifdef DEBUG
                        printf("\t[1]unvisited_SI[%2d] => %2x, dist[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], neighborNodeID, dist_arr[neighborNodeID]);
                        #endif
                    }
                    else if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){ //traverse to discovered successor and record its SI
                        sharedBitIndex[neighborNodeID] |= sharedBitIndex[currentNodeID];
                        
                        #ifdef DEBUG
                        printf("\t[2]visited_SI[%2d] => %2x, dist[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], neighborNodeID, dist_arr[neighborNodeID]);
                        #endif
                    }
                    else if(dist_arr[neighborNodeID] == dist_arr[currentNodeID]){ //traverse to discovered neighbor which is at same level as currentNodeID
                        relation[currentNodeID] |= sharedBitIndex[neighborNodeID] & (~sharedBitIndex[currentNodeID]);
                    
                        #ifdef DEBUG
                        printf("\t[3]Red edge found(%2d, %2d), ", currentNodeID, neighborNodeID);
                        printf("relation[%2d] = %2x\n", currentNodeID, relation[currentNodeID], neighborNodeID, relation[neighborNodeID]);
                        #endif
                    }
                }
            }
            
            //each sourceID update its CC with self ff, since dist_arr[sourceID] now is 0
            _CCs[sourceID] += _csr->ff[sourceID];

            #ifdef DEBUG
            printf("\n####      Source %2d 2nd traverse...      ####\n\n", sourceID);
            #endif

            Q->front = 0;
            while(!qIsEmpty(Q)){
                currentNodeID = qPopFront(Q);

                #ifdef DEBUG
                printf("currentNodeID = %2d ... dist = %2d ... relation = %x\n", currentNodeID, dist_arr[currentNodeID]);
                #endif

                for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborNodeID = _csr->csrE[neighborIndex];

                    if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){
                        relation[neighborNodeID] |= relation[currentNodeID];

                        #ifdef DEBUG
                        printf("\t[4]relation[%2d] = %2x\n", neighborNodeID, relation[neighborNodeID]);
                        #endif

                    }
                }
            }
            #pragma endregion //SourceTraverse



            #pragma region checkingSource_DistAns
            #ifdef CheckDistAns
            checkDistAns_exceptD1Node(_csr, dist_arr, sourceID);
            #endif

            #ifdef CheckCC_Ans
            dynamic_D1_CC_trace_component_Ans(_csr, _CCs, sourceID);
            #endif
            #pragma endregion //checkingSourceDistAns



            // #pragma region sourceCC_ComponentAccumulation_pushBased
            // //The remaining nodes in the component should be accumulate like below
            // for(int tempNotD1NodeIndex = 0 ; tempNotD1NodeIndex < _csr->ordinaryNodeCount ; tempNotD1NodeIndex ++){
            //     int notD1Node = _csr->notD1Node[tempNotD1NodeIndex];
            //     _CCs[notD1Node] += _csr->ff[sourceID] + dist_arr[notD1Node] * _csr->representNode[sourceID];
            // }
            // #pragma endregion //sourceCC_ComponentAccumulation_pushBased

            

            #pragma region neighborOfSource_GetDist_and_AccumulationCC
            //recover the distance from source to neighbor of source
            for(int sourceNeighborIndex = 0 ; sourceNeighborIndex < mappingCount ; sourceNeighborIndex ++){


                #pragma region GetDist
                memset(neighbor_dist_ans, -1, sizeof(int) * _csr->csrVSize);

                int sourceNeighborID = mapping_SI[sourceNeighborIndex];
                unsigned int bit_SI = 1 << sourceNeighborIndex;

                nodeDone[sourceNeighborID] = 1;
                
                #ifdef DEBUG
                printf("\nnextBFS = %2d, bit_SI = %x\n", sourceNeighborID, bit_SI);
                #endif
                
                for(int tempNotD1NodeIndex = 0 ; tempNotD1NodeIndex < _csr->ordinaryNodeCount ; tempNotD1NodeIndex ++){
                    int nD1NodeID = _csr->notD1Node[tempNotD1NodeIndex];

                    if((sharedBitIndex[nD1NodeID] & bit_SI) > 0){
                        neighbor_dist_ans[nD1NodeID] = dist_arr[nD1NodeID] - 1;
                    
                        #ifdef DEBUG
                        printf("\t[5]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", nD1NodeID, neighbor_dist_ans[nD1NodeID], nD1NodeID, sharedBitIndex[nD1NodeID]);
                        #endif

                    }
                    else{
                        neighbor_dist_ans[nD1NodeID] = dist_arr[nD1NodeID] + 1;
                        
                        #ifdef DEBUG
                        printf("\t[6]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", nD1NodeID, neighbor_dist_ans[nD1NodeID], nD1NodeID, sharedBitIndex[nD1NodeID]);
                        #endif

                        if((relation[nD1NodeID] & bit_SI) > 0){
                            neighbor_dist_ans[nD1NodeID] --;
                            
                            #ifdef DEBUG
                            printf("\t[7]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", nD1NodeID, neighbor_dist_ans[nD1NodeID], nD1NodeID, sharedBitIndex[nD1NodeID]);
                            #endif

                        }
                    }
                    
                    //update CC (push-based)
                    _CCs[nD1NodeID] += _csr->ff[sourceNeighborID] + neighbor_dist_ans[nD1NodeID] * _csr->representNode[sourceNeighborID];
                }

                #pragma region checkingSourceNeighbor_DistAns
                #ifdef CheckDistAns
                checkDistAns_exceptD1Node(_csr, neighbor_dist_ans, sourceNeighborID);
                #endif

                #ifdef CheckCC_Ans
                dynamic_D1_CC_trace_component_Ans(_csr, _CCs, sourceNeighborID);
                #endif
                #pragma endregion //checkingSourceNeighbor_DistAns

                #pragma endregion //GetDist


                // #pragma region sourceNeighborCC_ComponentAccumulation_pushBased
                // for(int tempNotD1NodeIndex = 0 ; tempNotD1NodeIndex < _csr->ordinaryNodeCount ; tempNotD1NodeIndex ++){
                //     int notD1Node = _csr->notD1Node[tempNotD1NodeIndex];
                //     _CCs[notD1Node] += _csr->ff[sourceNeighborID] + neighbor_dist_ans[notD1Node] * _csr->representNode[sourceNeighborID];
                // }
                // #pragma endregion //sourceNeighborCC_ComponentAccumulation_pushBased
            }
            #pragma endregion //neighborOfSource_GetDist_and_AccumulationCC

            //reset the SI & relation arrays
            memset(relation, 0, sizeof(unsigned int) * _csr->csrVSize);
            memset(sharedBitIndex, 0, sizeof(unsigned int) * _csr->csrVSize);

            // break;
        }
    }

    #pragma region d1GetCC_FromParent
    /**
     * Updating CC of d1Nodes is start from the last d1Node in _csr->degreeOneNodesQ,
     * since the nodes in rear end in _csr->degreeOneNodesQ are closer to the component,
     * the nodes in front end in _csr->degreeOneNodesQ are farther from the component
     * 
     * ##         Process d1Node from rear to front        ##
    */

    int d1NodeID        = -1;
    int d1NodeParentID  = -1;
    for(int d1NodeIndex = _csr->degreeOneNodesQ->rear ; d1NodeIndex >= 0 ; d1NodeIndex --){
        d1NodeID        = _csr->degreeOneNodesQ->dataArr[d1NodeIndex];
        d1NodeParentID  = _csr->D1Parent[d1NodeID];
        _CCs[d1NodeID]  = _CCs[d1NodeParentID] + _csr->totalNodeNumber - 2 * _csr->representNode[d1NodeID];
        
        #ifdef CheckCC_Ans
        dynamic_D1_CC_trace_D1_Ans(_csr, _CCs, d1NodeID);
        #endif
    }
    #pragma endregion //d1GetCC_FromParent
    printf("\n");
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

/**
 * @brief
 * Sort the nodeID with the order that from high degree to low degree,
 * then use the order we got to pick high degree node as source to traverse
*/
void compute_D1_CC_shareBased_DegreeOrder(struct CSR* _csr, int* _CCs){
    
    // double time1 = 0;
    // double time2 = 0;

    int* dist_arr           = (int*)malloc(sizeof(int) * _csr->csrVSize);
    int* neighbor_dist_ans  = (int*)malloc(sizeof(int) * _csr->csrVSize);

    struct qQueue* Q = InitqQueue();
    qInitResize(Q, _csr->csrVSize);

    //record that nodes which havn't been source yet
    int* nodeDone = (int*)calloc(sizeof(int), _csr->csrVSize);

    //record nodes belongs to which neighbor of source
    int* mapping_SI                 = (int*)malloc(sizeof(int) * 32);
    unsigned int* sharedBitIndex    = (unsigned int*)calloc(sizeof(unsigned int), _csr->csrVSize);
    unsigned int* relation          = (unsigned int*)calloc(sizeof(unsigned int), _csr->csrVSize);

    
    //Folding D1
    // time1 = seconds();
    D1Folding(_csr);
    // time2 = seconds();
    // printf("[Execution Time] D1Folding = %f\n", time2 - time1);
    /**
     * After D1Folding, we've got two lists which are:
     * 1. _csr->d1Node_List         : _csr->degreeOneNodesQ->dataArr
     * 2. _csr->notD1Node_List      : _csr->notD1Node 
    */

    //sorting by the order of degree from high to low
    // time1 = seconds();
    quicksort_nodeID_with_degree(_csr->notD1Node, _csr->csrNodesDegree, 0, _csr->ordinaryNodeCount - 1);
    // time2 = seconds();
    // printf("[Execution Time] quickSort = %f\n", time2 - time1);

    #pragma region Traverse
    int ordinaryTraversalCount  = 0;
    int sharingTraversalCount   = 0;
    int sourceID = -1;
    for(int notD1NodeIndex = _csr->ordinaryNodeCount - 1 ; notD1NodeIndex >= 0 ; notD1NodeIndex --){
        sourceID = _csr->notD1Node[notD1NodeIndex];
        if(nodeDone[sourceID] == 1){
            continue;
        }

        nodeDone[sourceID] = 1;

        #ifdef DEBUG
        printf("\nsourceID = %2d, degree[%2d] = %2d ...\n", sourceID, sourceID, _csr->csrNodesDegree[sourceID]);
        #endif

        //reset dist_arr        
        memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);

        //reset Queue
        resetQueue(Q);

        //Init dist_arr[sourceID]
        dist_arr[sourceID] = 0;

        //push sourceID into Q
        qPushBack(Q, sourceID);

        register int currentNodeID  = -1;
        register int neighborNodeID = -1;
        register int neighborIndex  = -1;

        // Count neighbors of sourceID which are not done yet
        int mappingCount = 0;
        for(int neighborIndex = _csr->csrV[sourceID] ; neighborIndex < _csr->oriCsrV[sourceID + 1] ; neighborIndex ++){
            neighborNodeID = _csr->csrE[neighborIndex];
            if(nodeDone[neighborNodeID] == 0){
                mappingCount ++;
            }
        }

        //decide to use the sharing strategy or not
        if(mappingCount < 3){ // perform ordinary traverse
            ordinaryTraversalCount ++;
            
            while(!qIsEmpty(Q)){
                currentNodeID = qPopFront(Q);

                for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborNodeID = _csr->csrE[neighborIndex];
                    
                    if(dist_arr[neighborNodeID] == -1){
                        qPushBack(Q, neighborNodeID);
                        dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;

                        _CCs[neighborNodeID] += _csr->ff[sourceID] + dist_arr[neighborNodeID] * _csr->representNode[sourceID];
                    }
                }
            }
            _CCs[sourceID] += _csr->ff[sourceID];

        }
        else{ //perform traversal with sharing strategy
            sharingTraversalCount ++;

            #pragma region mappingNeighbor
            mappingCount = 0;
            //each neighbor of sourceID mapping to SBI, if it haven't been source yet
            for(neighborIndex = _csr->csrV[sourceID] ; neighborIndex < _csr->oriCsrV[sourceID + 1] ; neighborIndex ++){
                neighborNodeID = _csr->csrE[neighborIndex];

                if(nodeDone[neighborNodeID] == 0){
                    
                    sharedBitIndex[neighborNodeID] = 1 << mappingCount;
                    mapping_SI[mappingCount] = neighborNodeID;
                    
                    #ifdef DEBUG
                    printf("sharedBitIndex[%2d] = %8x, \tmapping_SI[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], mappingCount, mapping_SI[mappingCount]);
                    #endif

                    mappingCount ++;

                    //Record to 32 bit only
                    if(mappingCount == 32){
                        break;
                    }
                    
                }
            }
            #pragma endregion //mappingNeighbor
            


            #pragma region SourceTraverse
            /**
             * Main source traverse for getting some information:
             * 
             * ##        First Traverse        ##
             * 1. dist              : distance from source to each node in the component
             * 2. sharedBitIndex    : record each node that can be found first by which neighbor of source
             * ##################################
             * 
             * ##        Second Traverse        ##
             * 3. relation          : record each node that should remain the distance when source are some specific nodes
             * ###################################
             * 
             * @todo 
             * 1. consider another way to replace "the 3 if statement", the way of branchless technique may be a good choice
             * 2. A way : when currentNodeID find unvisited node v, then just push v into Q, and don't update its dist,
             *          until v becomes currentNodeID, then assign the distance for v.
            */

            #ifdef DEBUG
            printf("\n####      Source %2d 1st traverse...      ####\n\n", sourceID);
            #endif

            while(!qIsEmpty(Q)){
                currentNodeID = qPopFront(Q);
                
                #ifdef DEBUG
                printf("currentNodeID = %2d ... dist = %2d ... SI = %x ... relation = %x\n", currentNodeID, dist_arr[currentNodeID], sharedBitIndex[currentNodeID], relation[currentNodeID]);
                #endif

                for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborNodeID = _csr->csrE[neighborIndex];

                    if(dist_arr[neighborNodeID] == -1){ //traverse new succesor and update its SI
                        qPushBack(Q, neighborNodeID);
                        dist_arr[neighborNodeID]        = dist_arr[currentNodeID] + 1;
                        sharedBitIndex[neighborNodeID] |= sharedBitIndex[currentNodeID];
                        
                        //update CC (push-based)
                        _CCs[neighborNodeID] += _csr->ff[sourceID] + dist_arr[neighborNodeID] * _csr->representNode[sourceID];

                        #ifdef DEBUG
                        printf("\t[1]unvisited_SI[%2d] => %2x, dist[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], neighborNodeID, dist_arr[neighborNodeID]);
                        #endif
                    }
                    else if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){ //traverse to discovered successor and record its SI
                        sharedBitIndex[neighborNodeID] |= sharedBitIndex[currentNodeID];
                        
                        #ifdef DEBUG
                        printf("\t[2]visited_SI[%2d] => %2x, dist[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], neighborNodeID, dist_arr[neighborNodeID]);
                        #endif
                    }
                    else if(dist_arr[neighborNodeID] == dist_arr[currentNodeID]){ //traverse to discovered neighbor which is at same level as currentNodeID
                        relation[currentNodeID] |= sharedBitIndex[neighborNodeID] & (~sharedBitIndex[currentNodeID]);
                    
                        #ifdef DEBUG
                        printf("\t[3]Red edge found(%2d, %2d), ", currentNodeID, neighborNodeID);
                        printf("relation[%2d] = %2x\n", currentNodeID, relation[currentNodeID], neighborNodeID, relation[neighborNodeID]);
                        #endif
                    }
                }
            }
            
            //each sourceID update its CC with self ff, since dist_arr[sourceID] now is 0
            _CCs[sourceID] += _csr->ff[sourceID];

            #ifdef DEBUG
            printf("\n####      Source %2d 2nd traverse...      ####\n\n", sourceID);
            #endif

            Q->front = 0;
            while(!qIsEmpty(Q)){
                currentNodeID = qPopFront(Q);

                #ifdef DEBUG
                printf("currentNodeID = %2d ... dist = %2d ... relation = %x\n", currentNodeID, dist_arr[currentNodeID]);
                #endif

                for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborNodeID = _csr->csrE[neighborIndex];

                    if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){
                        relation[neighborNodeID] |= relation[currentNodeID];

                        #ifdef DEBUG
                        printf("\t[4]relation[%2d] = %2x\n", neighborNodeID, relation[neighborNodeID]);
                        #endif

                    }
                }
            }
            #pragma endregion //SourceTraverse



            #pragma region checkingSource_DistAns
            #ifdef CheckDistAns
            checkDistAns_exceptD1Node(_csr, dist_arr, sourceID);
            #endif

            #ifdef CheckCC_Ans
            dynamic_D1_CC_trace_component_Ans(_csr, _CCs, sourceID);
            #endif
            #pragma endregion //checkingSourceDistAns



            // #pragma region sourceCC_ComponentAccumulation_pushBased
            // //The remaining nodes in the component should be accumulate like below
            // for(int tempNotD1NodeIndex = 0 ; tempNotD1NodeIndex < _csr->ordinaryNodeCount ; tempNotD1NodeIndex ++){
            //     int notD1Node = _csr->notD1Node[tempNotD1NodeIndex];
            //     _CCs[notD1Node] += _csr->ff[sourceID] + dist_arr[notD1Node] * _csr->representNode[sourceID];
            // }
            // #pragma endregion //sourceCC_ComponentAccumulation_pushBased

            

            #pragma region neighborOfSource_GetDist_and_AccumulationCC
            //recover the distance from source to neighbor of source
            for(int sourceNeighborIndex = 0 ; sourceNeighborIndex < mappingCount ; sourceNeighborIndex ++){


                #pragma region GetDist
                memset(neighbor_dist_ans, -1, sizeof(int) * _csr->csrVSize);

                int sourceNeighborID = mapping_SI[sourceNeighborIndex];
                unsigned int bit_SI = 1 << sourceNeighborIndex;

                nodeDone[sourceNeighborID] = 1;
                
                #ifdef DEBUG
                printf("\nnextBFS = %2d, bit_SI = %x\n", sourceNeighborID, bit_SI);
                #endif
                
                for(int tempNotD1NodeIndex = 0 ; tempNotD1NodeIndex < _csr->ordinaryNodeCount ; tempNotD1NodeIndex ++){
                    int nD1NodeID = _csr->notD1Node[tempNotD1NodeIndex];

                    if((sharedBitIndex[nD1NodeID] & bit_SI) > 0){
                        neighbor_dist_ans[nD1NodeID] = dist_arr[nD1NodeID] - 1;
                    
                        #ifdef DEBUG
                        printf("\t[5]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", nD1NodeID, neighbor_dist_ans[nD1NodeID], nD1NodeID, sharedBitIndex[nD1NodeID]);
                        #endif

                    }
                    else{
                        neighbor_dist_ans[nD1NodeID] = dist_arr[nD1NodeID] + 1;
                        
                        #ifdef DEBUG
                        printf("\t[6]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", nD1NodeID, neighbor_dist_ans[nD1NodeID], nD1NodeID, sharedBitIndex[nD1NodeID]);
                        #endif

                        if((relation[nD1NodeID] & bit_SI) > 0){
                            neighbor_dist_ans[nD1NodeID] --;
                            
                            #ifdef DEBUG
                            printf("\t[7]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", nD1NodeID, neighbor_dist_ans[nD1NodeID], nD1NodeID, sharedBitIndex[nD1NodeID]);
                            #endif

                        }
                    }
                    
                    //update CC (push-based)
                    _CCs[nD1NodeID] += _csr->ff[sourceNeighborID] + neighbor_dist_ans[nD1NodeID] * _csr->representNode[sourceNeighborID];
                }

                #pragma region checkingSourceNeighbor_DistAns
                #ifdef CheckDistAns
                checkDistAns_exceptD1Node(_csr, neighbor_dist_ans, sourceNeighborID);
                #endif

                #ifdef CheckCC_Ans
                dynamic_D1_CC_trace_component_Ans(_csr, _CCs, sourceNeighborID);
                #endif
                #pragma endregion //checkingSourceNeighbor_DistAns

                #pragma endregion //GetDist


                // #pragma region sourceNeighborCC_ComponentAccumulation_pushBased
                // for(int tempNotD1NodeIndex = 0 ; tempNotD1NodeIndex < _csr->ordinaryNodeCount ; tempNotD1NodeIndex ++){
                //     int notD1Node = _csr->notD1Node[tempNotD1NodeIndex];
                //     _CCs[notD1Node] += _csr->ff[sourceNeighborID] + neighbor_dist_ans[notD1Node] * _csr->representNode[sourceNeighborID];
                // }
                // #pragma endregion //sourceNeighborCC_ComponentAccumulation_pushBased
            }
            #pragma endregion //neighborOfSource_GetDist_and_AccumulationCC

            //reset the SI & relation arrays
            memset(relation, 0, sizeof(unsigned int) * _csr->csrVSize);
            memset(sharedBitIndex, 0, sizeof(unsigned int) * _csr->csrVSize);

            // break;
        }
    }
    #pragma endregion //Traverse



    #pragma region d1GetCC_FromParent
    /**
     * Updating CC of d1Nodes is start from the last d1Node in _csr->degreeOneNodesQ,
     * since the nodes in rear end in _csr->degreeOneNodesQ are closer to the component,
     * the nodes in front end in _csr->degreeOneNodesQ are farther from the component
     * 
     * ##         Process d1Node from rear to front        ##
    */

    int d1NodeID        = -1;
    int d1NodeParentID  = -1;
    for(int d1NodeIndex = _csr->degreeOneNodesQ->rear ; d1NodeIndex >= 0 ; d1NodeIndex --){
        d1NodeID        = _csr->degreeOneNodesQ->dataArr[d1NodeIndex];
        d1NodeParentID  = _csr->D1Parent[d1NodeID];
        _CCs[d1NodeID]  = _CCs[d1NodeParentID] + _csr->totalNodeNumber - 2 * _csr->representNode[d1NodeID];
        
        #ifdef CheckCC_Ans
        dynamic_D1_CC_trace_D1_Ans(_csr, _CCs, d1NodeID);
        #endif
    }
    #pragma endregion //d1GetCC_FromParent
    printf("\n");
}

/**
 * @brief
 * Just 64-bit SI & 64-bit relation version of compute_D1_CC_shareBased_DegreeOrder
*/
void compute_D1_CC_sharedBased_DegreeOrder_64bit(struct CSR* _csr, int* _CCs){
    // double time1 = 0;
    // double time2 = 0;

    int* dist_arr           = (int*)malloc(sizeof(int) * _csr->csrVSize);
    int* neighbor_dist_ans  = (int*)malloc(sizeof(int) * _csr->csrVSize);

    struct qQueue* Q = InitqQueue();
    qInitResize(Q, _csr->csrVSize);

    //record that nodes which havn't been source yet
    int* nodeDone = (int*)calloc(sizeof(int), _csr->csrVSize);

    //record nodes belongs to which neighbor of source
    int* mapping_SI                     = (int*)malloc(sizeof(int) * 64);
    unsigned long long* sharedBitIndex  = (unsigned long long*)calloc(sizeof(unsigned long long), _csr->csrVSize);
    unsigned long long* relation        = (unsigned long long*)calloc(sizeof(unsigned long long), _csr->csrVSize);

    
    //Folding D1
    // time1 = seconds();
    D1Folding(_csr);
    // time2 = seconds();
    // printf("[Execution Time] D1Folding = %f\n", time2 - time1);
    /**
     * After D1Folding, we've got two lists which are:
     * 1. _csr->d1Node_List         : _csr->degreeOneNodesQ->dataArr
     * 2. _csr->notD1Node_List      : _csr->notD1Node 
    */

    //sorting by the order of degree from high to low
    // time1 = seconds();
    quicksort_nodeID_with_degree(_csr->notD1Node, _csr->csrNodesDegree, 0, _csr->ordinaryNodeCount - 1);
    // time2 = seconds();
    // printf("[Execution Time] quickSort = %f\n", time2 - time1);

    #pragma region Traverse
    // int ordinaryTraversalCount  = 0;
    // int sharingTraversalCount   = 0;
    int sourceID = -1;
    for(int notD1NodeIndex = _csr->ordinaryNodeCount - 1 ; notD1NodeIndex >= 0 ; notD1NodeIndex --){
        sourceID = _csr->notD1Node[notD1NodeIndex];
        if(nodeDone[sourceID] == 1){
            continue;
        }

        nodeDone[sourceID] = 1;

        #ifdef DEBUG
        printf("\nsourceID = %2d, degree[%2d] = %2d ...\n", sourceID, sourceID, _csr->csrNodesDegree[sourceID]);
        #endif

        //reset dist_arr        
        memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);

        //reset Queue
        resetQueue(Q);

        //Init dist_arr[sourceID]
        dist_arr[sourceID] = 0;

        //push sourceID into Q
        qPushBack(Q, sourceID);

        register int currentNodeID  = -1;
        register int neighborNodeID = -1;
        register int neighborIndex  = -1;

        // Count neighbors of sourceID which are not done yet
        int mappingCount = 0;
        for(int neighborIndex = _csr->csrV[sourceID] ; neighborIndex < _csr->oriCsrV[sourceID + 1] ; neighborIndex ++){
            neighborNodeID = _csr->csrE[neighborIndex];
            if(nodeDone[neighborNodeID] == 0){
                mappingCount ++;
            }
        }

        //decide to use the sharing strategy or not
        if(mappingCount < 3){ // perform ordinary traverse
            // ordinaryTraversalCount ++;
            
            while(!qIsEmpty(Q)){
                currentNodeID = qPopFront(Q);

                for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborNodeID = _csr->csrE[neighborIndex];
                    
                    if(dist_arr[neighborNodeID] == -1){
                        qPushBack(Q, neighborNodeID);
                        dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;

                        _CCs[neighborNodeID] += _csr->ff[sourceID] + dist_arr[neighborNodeID] * _csr->representNode[sourceID];
                    }
                }
            }
            _CCs[sourceID] += _csr->ff[sourceID];

        }
        else{ //perform traversal with sharing strategy
            // sharingTraversalCount ++;

            #pragma region mappingNeighbor
            mappingCount = 0;
            //each neighbor of sourceID mapping to SBI, if it haven't been source yet
            for(neighborIndex = _csr->csrV[sourceID] ; neighborIndex < _csr->oriCsrV[sourceID + 1] ; neighborIndex ++){
                neighborNodeID = _csr->csrE[neighborIndex];

                if(nodeDone[neighborNodeID] == 0){
                    
                    sharedBitIndex[neighborNodeID] = 1 << mappingCount;
                    mapping_SI[mappingCount] = neighborNodeID;
                    
                    #ifdef DEBUG
                    printf("sharedBitIndex[%2d] = %8x, \tmapping_SI[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], mappingCount, mapping_SI[mappingCount]);
                    #endif

                    mappingCount ++;

                    //Record to 32 bit only
                    if(mappingCount == 64){
                        break;
                    }
                    
                }
            }
            #pragma endregion //mappingNeighbor
            


            #pragma region SourceTraverse
            /**
             * Main source traverse for getting some information:
             * 
             * ##        First Traverse        ##
             * 1. dist              : distance from source to each node in the component
             * 2. sharedBitIndex    : record each node that can be found first by which neighbor of source
             * ##################################
             * 
             * ##        Second Traverse        ##
             * 3. relation          : record each node that should remain the distance when source are some specific nodes
             * ###################################
             * 
             * @todo 
             * 1. consider another way to replace "the 3 if statement", the way of branchless technique may be a good choice
             * 2. A way : when currentNodeID find unvisited node v, then just push v into Q, and don't update its dist,
             *          until v becomes currentNodeID, then assign the distance for v.
            */

            #ifdef DEBUG
            printf("\n####      Source %2d 1st traverse...      ####\n\n", sourceID);
            #endif

            while(!qIsEmpty(Q)){
                currentNodeID = qPopFront(Q);
                
                #ifdef DEBUG
                printf("currentNodeID = %2d ... dist = %2d ... SI = %x ... relation = %x\n", currentNodeID, dist_arr[currentNodeID], sharedBitIndex[currentNodeID], relation[currentNodeID]);
                #endif

                for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborNodeID = _csr->csrE[neighborIndex];

                    if(dist_arr[neighborNodeID] == -1){ //traverse new succesor and update its SI
                        qPushBack(Q, neighborNodeID);
                        dist_arr[neighborNodeID]        = dist_arr[currentNodeID] + 1;
                        sharedBitIndex[neighborNodeID] |= sharedBitIndex[currentNodeID];
                        
                        //update CC (push-based)
                        _CCs[neighborNodeID] += _csr->ff[sourceID] + dist_arr[neighborNodeID] * _csr->representNode[sourceID];

                        #ifdef DEBUG
                        printf("\t[1]unvisited_SI[%2d] => %2x, dist[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], neighborNodeID, dist_arr[neighborNodeID]);
                        #endif
                    }
                    else if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){ //traverse to discovered successor and record its SI
                        sharedBitIndex[neighborNodeID] |= sharedBitIndex[currentNodeID];
                        
                        #ifdef DEBUG
                        printf("\t[2]visited_SI[%2d] => %2x, dist[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], neighborNodeID, dist_arr[neighborNodeID]);
                        #endif
                    }
                    else if(dist_arr[neighborNodeID] == dist_arr[currentNodeID]){ //traverse to discovered neighbor which is at same level as currentNodeID
                        relation[currentNodeID] |= sharedBitIndex[neighborNodeID] & (~sharedBitIndex[currentNodeID]);
                    
                        #ifdef DEBUG
                        printf("\t[3]Red edge found(%2d, %2d), ", currentNodeID, neighborNodeID);
                        printf("relation[%2d] = %2x\n", currentNodeID, relation[currentNodeID], neighborNodeID, relation[neighborNodeID]);
                        #endif
                    }
                }
            }
            
            //each sourceID update its CC with self ff, since dist_arr[sourceID] now is 0
            _CCs[sourceID] += _csr->ff[sourceID];

            #ifdef DEBUG
            printf("\n####      Source %2d 2nd traverse...      ####\n\n", sourceID);
            #endif

            Q->front = 0;
            while(!qIsEmpty(Q)){
                currentNodeID = qPopFront(Q);

                #ifdef DEBUG
                printf("currentNodeID = %2d ... dist = %2d ... relation = %x\n", currentNodeID, dist_arr[currentNodeID]);
                #endif

                for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborNodeID = _csr->csrE[neighborIndex];

                    if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){
                        relation[neighborNodeID] |= relation[currentNodeID];

                        #ifdef DEBUG
                        printf("\t[4]relation[%2d] = %2x\n", neighborNodeID, relation[neighborNodeID]);
                        #endif

                    }
                }
            }
            #pragma endregion //SourceTraverse



            #pragma region checkingSource_DistAns
            #ifdef CheckDistAns
            checkDistAns_exceptD1Node(_csr, dist_arr, sourceID);
            #endif

            #ifdef CheckCC_Ans
            dynamic_D1_CC_trace_component_Ans(_csr, _CCs, sourceID);
            #endif
            #pragma endregion //checkingSourceDistAns



            // #pragma region sourceCC_ComponentAccumulation_pushBased
            // //The remaining nodes in the component should be accumulate like below
            // for(int tempNotD1NodeIndex = 0 ; tempNotD1NodeIndex < _csr->ordinaryNodeCount ; tempNotD1NodeIndex ++){
            //     int notD1Node = _csr->notD1Node[tempNotD1NodeIndex];
            //     _CCs[notD1Node] += _csr->ff[sourceID] + dist_arr[notD1Node] * _csr->representNode[sourceID];
            // }
            // #pragma endregion //sourceCC_ComponentAccumulation_pushBased

            

            #pragma region neighborOfSource_GetDist_and_AccumulationCC
            //recover the distance from source to neighbor of source
            for(int sourceNeighborIndex = 0 ; sourceNeighborIndex < mappingCount ; sourceNeighborIndex ++){


                #pragma region GetDist
                memset(neighbor_dist_ans, -1, sizeof(int) * _csr->csrVSize);

                int sourceNeighborID = mapping_SI[sourceNeighborIndex];
                unsigned long long bit_SI = 1 << sourceNeighborIndex;

                nodeDone[sourceNeighborID] = 1;
                
                #ifdef DEBUG
                printf("\nnextBFS = %2d, bit_SI = %x\n", sourceNeighborID, bit_SI);
                #endif
                
                for(int tempNotD1NodeIndex = 0 ; tempNotD1NodeIndex < _csr->ordinaryNodeCount ; tempNotD1NodeIndex ++){
                    int nD1NodeID = _csr->notD1Node[tempNotD1NodeIndex];

                    if((sharedBitIndex[nD1NodeID] & bit_SI) > 0){
                        neighbor_dist_ans[nD1NodeID] = dist_arr[nD1NodeID] - 1;
                    
                        #ifdef DEBUG
                        printf("\t[5]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", nD1NodeID, neighbor_dist_ans[nD1NodeID], nD1NodeID, sharedBitIndex[nD1NodeID]);
                        #endif

                    }
                    else{
                        neighbor_dist_ans[nD1NodeID] = dist_arr[nD1NodeID] + 1;
                        
                        #ifdef DEBUG
                        printf("\t[6]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", nD1NodeID, neighbor_dist_ans[nD1NodeID], nD1NodeID, sharedBitIndex[nD1NodeID]);
                        #endif

                        if((relation[nD1NodeID] & bit_SI) > 0){
                            neighbor_dist_ans[nD1NodeID] --;
                            
                            #ifdef DEBUG
                            printf("\t[7]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", nD1NodeID, neighbor_dist_ans[nD1NodeID], nD1NodeID, sharedBitIndex[nD1NodeID]);
                            #endif

                        }
                    }
                    
                    //update CC (push-based)
                    _CCs[nD1NodeID] += _csr->ff[sourceNeighborID] + neighbor_dist_ans[nD1NodeID] * _csr->representNode[sourceNeighborID];
                }

                #pragma region checkingSourceNeighbor_DistAns
                #ifdef CheckDistAns
                checkDistAns_exceptD1Node(_csr, neighbor_dist_ans, sourceNeighborID);
                #endif

                #ifdef CheckCC_Ans
                dynamic_D1_CC_trace_component_Ans(_csr, _CCs, sourceNeighborID);
                #endif
                #pragma endregion //checkingSourceNeighbor_DistAns

                #pragma endregion //GetDist


                // #pragma region sourceNeighborCC_ComponentAccumulation_pushBased
                // for(int tempNotD1NodeIndex = 0 ; tempNotD1NodeIndex < _csr->ordinaryNodeCount ; tempNotD1NodeIndex ++){
                //     int notD1Node = _csr->notD1Node[tempNotD1NodeIndex];
                //     _CCs[notD1Node] += _csr->ff[sourceNeighborID] + neighbor_dist_ans[notD1Node] * _csr->representNode[sourceNeighborID];
                // }
                // #pragma endregion //sourceNeighborCC_ComponentAccumulation_pushBased
            }
            #pragma endregion //neighborOfSource_GetDist_and_AccumulationCC

            //reset the SI & relation arrays
            memset(relation, 0, sizeof(unsigned long long) * _csr->csrVSize);
            memset(sharedBitIndex, 0, sizeof(unsigned long long) * _csr->csrVSize);

            // break;
        }
    }
    #pragma endregion //Traverse



    #pragma region d1GetCC_FromParent
    /**
     * Updating CC of d1Nodes is start from the last d1Node in _csr->degreeOneNodesQ,
     * since the nodes in rear end in _csr->degreeOneNodesQ are closer to the component,
     * the nodes in front end in _csr->degreeOneNodesQ are farther from the component
     * 
     * ##         Process d1Node from rear to front        ##
    */

    int d1NodeID        = -1;
    int d1NodeParentID  = -1;
    for(int d1NodeIndex = _csr->degreeOneNodesQ->rear ; d1NodeIndex >= 0 ; d1NodeIndex --){
        d1NodeID        = _csr->degreeOneNodesQ->dataArr[d1NodeIndex];
        d1NodeParentID  = _csr->D1Parent[d1NodeID];
        _CCs[d1NodeID]  = _CCs[d1NodeParentID] + _csr->totalNodeNumber - 2 * _csr->representNode[d1NodeID];
        
        #ifdef CheckCC_Ans
        dynamic_D1_CC_trace_D1_Ans(_csr, _CCs, d1NodeID);
        #endif
    }
    #pragma endregion //d1GetCC_FromParent
    printf("\n");
}


void compute_D1_AP_CC(struct CSR* _csr, int* _CCs){
    int* dist_arr = (int*)malloc(sizeof(int) * _csr->csrVSize * 2);
    struct qQueue* Q = InitqQueue();
    qInitResize(Q, _csr->csrVSize * 2);

    //D1 Folding
    D1Folding(_csr);

    //AP Process
    AP_detection(_csr);
    AP_Copy_And_Split(_csr);

    int oriEndNodeID = _csr->endNodeID - _csr->apCloneCount;
    printf("oriEndNodeID = %d\n", oriEndNodeID);

    for(int sourceID = _csr->startNodeID ; sourceID <= oriEndNodeID ; sourceID ++){
        if(_csr->nodesType[sourceID] & D1 || _csr->nodesType[sourceID] & OriginAP){
            continue;
        }

        //reset Q
        Q->front = 0;
        Q->rear = -1;
        memset(dist_arr, -1, sizeof(int) * _csr->csrVSize * 2);

        //Init
        dist_arr[sourceID] = 0;
        qPushBack(Q, sourceID);
        
        //traverse
        register int curID  = -1;
        register int nid    = -1;
        register int nidx   = -1;
        while(!qIsEmpty(Q)){
            curID = qPopFront(Q);
            for(nidx = _csr->csrV[curID] ; nidx < _csr->oriCsrV[curID + 1] ; nidx ++){
                nid = _csr->csrE[nidx];

                if(dist_arr[nid] == -1){
                    qPushBack(Q, nid);
                    dist_arr[nid] = dist_arr[curID] + 1;
                    
                    _CCs[sourceID] += _csr->ff[nid] + dist_arr[nid] * _csr->representNode[nid];
                }
            }
        }
        _CCs[sourceID] += _csr->ff[sourceID];
    }

    #pragma region d1Node_Dist_And_CC_Recovery
    // printf("_csr->totalNodeNumber = %2d\n", _csr->totalNodeNumber);
    int d1NodeID        = -1;
    int d1NodeParentID  = -1;
    for(int d1NodeIndex = _csr->degreeOneNodesQ->rear ; d1NodeIndex >= 0 ; d1NodeIndex --){
        d1NodeID        = _csr->degreeOneNodesQ->dataArr[d1NodeIndex];
        d1NodeParentID  = _csr->D1Parent[d1NodeID];
        _CCs[d1NodeID]  = _CCs[d1NodeParentID] + _csr->totalNodeNumber - 2 * _csr->representNode[d1NodeID];
        // printf("d1NodeID = %2d, _CCs[%2d] = %2d, ParentID = %2d, _CCs[%2d] = %2d\n", d1NodeID, d1NodeID, _CCs[d1NodeID], d1NodeParentID, d1NodeParentID, _CCs[d1NodeParentID]);
    }
    #pragma endregion //d1Node_Dist_And_CC_Recovery

    // for(int ID = _csr->startNodeID ; ID <= oriEndNodeID ; ID ++){
    //     printf("CC[%d] = %d\n", ID, _CCs[ID]);
    // }
}

void compute_D1_AP_CC_shareBased_DegreeOrder(struct CSR* _csr, int* _CCs){
    int* dist_arr = (int*)malloc(sizeof(int) * (_csr->csrVSize) * 2);
    int* neighbor_dist_ans = (int*)malloc(sizeof(int) * (_csr->csrVSize) * 2);
    struct qQueue* Q = InitqQueue();
    qInitResize(Q, (_csr->csrVSize) * 2);

    //record that nodes which havn't been source yet
    int* nodeDone = (int*)calloc(sizeof(int*), (_csr->csrVSize) * 2);

    //record nodes belongs to which neighbor of source
    int* mapping_SI                 = (int*)malloc(sizeof(int) * 32);
    unsigned int* sharedBitIndex    = (unsigned int*)calloc(sizeof(unsigned int), (_csr->csrVSize) * 2);
    unsigned int* relation          = (unsigned int*)calloc(sizeof(unsigned int), (_csr->csrVSize) * 2);

    //D1 Folding
    D1Folding(_csr);

    //AP Process
    AP_detection(_csr);
    AP_Copy_And_Split(_csr);
    compactNodesByComp(_csr);
    //Sort aliveNodeID with degree
    quicksort_nodeID_with_degree(_csr->aliveNode, _csr->csrNodesDegree, 0, _csr->aliveNodeCount - 1);

    printf("[Check point 1]\n");

    for(int aliveNodeIndex = _csr->aliveNodeCount - 1 ; aliveNodeIndex >= 0 ; aliveNodeIndex --){
        int sourceID = _csr->aliveNode[aliveNodeIndex];

        printf("sourceID = %d, ", sourceID);

        if(nodeDone[sourceID] == 1){
            continue;
        }
        
        nodeDone[sourceID] = 1;

        //reset Q, dist_arr
        memset(dist_arr, -1, sizeof(int) * (_csr->csrVSize) * 2);
        resetQueue(Q);

        //Init Q, dist_arr
        dist_arr[sourceID] = 0;
        qPushBack(Q, sourceID);

        register int curID  = -1;
        register int nID    = -1;
        register int nidx   = -1;

        int mappingCount = 0;
        for(nidx = _csr->csrV[sourceID] ; nidx < _csr->oriCsrV[sourceID + 1] ; nidx ++){
            nID = _csr->csrE[nidx];
            if((nodeDone[nID] == 0) && (!(_csr->nodesType[nID] & ClonedAP)) && (!(_csr->nodesType[nID] & OriginAP))){
                mappingCount ++;
            }
        }
        printf("mappingCount = %d\n", mappingCount);

        //decide to use the sharing strategy or not
        if(mappingCount < 3){ //perform ordinary traverse
            while(!qIsEmpty(Q)){
                curID = qPopFront(Q);
                for(nidx = _csr->csrV[curID] ; nidx < _csr->oriCsrV[curID] ; nidx ++){
                    nID = _csr->csrE[nidx];

                    if(dist_arr[nID] == -1){
                        qPushBack(Q, nID);
                        dist_arr[nID] = dist_arr[curID] + 1;

                        _CCs[sourceID] += _csr->ff[nID] + dist_arr[nID] * _csr->representNode[nID];
                    }
                }
            }
            _CCs[sourceID] += _csr->ff[sourceID];
        }
        else{ //perform traversal with sharing strategy
            #pragma region mappingNeighbor
            mappingCount = 0;
            //each neighbor of sourceID mapping to SBI, if it haven't been sourceID and not ClonedAP
            for(nidx = _csr->csrV[sourceID] ; nidx < _csr->oriCsrV[sourceID + 1] ; nidx ++){
                nID = _csr->csrE[nidx];

                if(nodeDone[nID] == 0 && (!(_csr->nodesType[nID] & ClonedAP)) && (!(_csr->nodesType[nID] & OriginAP))){
                    sharedBitIndex[nID] = 1 << mappingCount;
                    mapping_SI[mappingCount] = nID;

                    // printf("\tshared nID %d\n", nID);

                    mappingCount ++;

                    //Record to 32 bit only
                    if(mappingCount == 32){
                        break;
                    }
                }
            }
            #pragma endregion mappingNeighbor

            #pragma region SourceTraverse
            /**
             * Main source traverse for getting some information:
             * 
             * ##        First Traverse        ##
             * 1. dist              : distance from source to each node in the component
             * 2. sharedBitIndex    : record each node that can be found first by which neighbor of source
             * ##################################
             * 
             * ##        Second Traverse        ##
             * 3. relation          : record each node that should remain the distance when source are some specific nodes
             * ###################################
             * 
             * @todo 
             * 1. consider another way to replace "the 3 if statement", the way of branchless technique may be a good choice
             * 2. A way : when currentNodeID find unvisited node v, then just push v into Q, and don't update its dist,
             *          until v becomes currentNodeID, then assign the distance for v.
            */
            
            printf("\n\t####      Source %2d 1st Traverse...      ####\n\n", sourceID);
            while(!qIsEmpty(Q)){
                curID = qPopFront(Q);
                
                printf("\tcurID = %2d, dist = %2d, SI = %x, relation = %x\n", curID, dist_arr[curID], sharedBitIndex[curID], relation[curID]);

                for(nidx = _csr->csrV[curID] ; nidx < _csr->oriCsrV[curID + 1] ; nidx ++){
                    nID = _csr->csrE[nidx];

                    if(dist_arr[nID] == -1){
                        qPushBack(Q, nID);
                        dist_arr[nID] = dist_arr[curID] + 1;
                        sharedBitIndex[nID] |= sharedBitIndex[curID];

                        _CCs[sourceID] += _csr->ff[nID] + dist_arr[nID] * _csr->representNode[nID];
                    }
                    else if(dist_arr[nID] == dist_arr[curID] + 1){
                        sharedBitIndex[nID] |= sharedBitIndex[curID];
                    }
                    else if(dist_arr[nID] == dist_arr[curID]){
                        relation[curID] |= sharedBitIndex[nID] & (~sharedBitIndex[curID]);
                    }
                }
            }
            _CCs[sourceID] += _csr->ff[sourceID];

            printf("\n\t####      Source %2d 2nd Traverse...      ####\n\n", sourceID);
            Q->front = 0;
            while(!qIsEmpty(Q)){
                curID = qPopFront(Q);

                for(nidx = _csr->csrV[curID] ; nidx < _csr->oriCsrV[curID + 1] ; nidx ++){
                    nID = _csr->csrE[nidx];

                    if(dist_arr[nID] == dist_arr[curID] + 1){
                        relation[nID] |= relation[curID];

                        printf("\trelation[%2d] = %2x\n", nID, relation[nID]);
                    }
                }
            }
            #pragma endregion SourceTraverse
        
            /**
             * @todo 
             * 把相同 component 的 nodeID 排在一起，用類似 csrV，csrE的方式記錄 component 的 nodeID，
             * 所以在這邊把 distance 復原的時候 只要復原一部分就可以了
            */
            #pragma region neighborOfSource_GetDist_and_AccumulationCC
            //recover the distance from source to neighbor of source
            for(int sourceNeighborIndex = 0 ; sourceNeighborIndex < mappingCount ; sourceNeighborIndex ++){
                // memset(neighbor_dist_ans, -1, sizeof(int) * _csr->csrVSize );
                int sourceNeighborID = mapping_SI[sourceNeighborIndex];
                unsigned int bit_SI = 1 << sourceNeighborIndex;
                
                
                nodeDone[sourceNeighborID] = 1;
                
                int nodeCompID = _csr->nodesCompID[sourceNeighborID];
                printf("\nnextBFS = %2d, bit_SI = %x, nodeCompID = %d\n", sourceNeighborID, bit_SI, nodeCompID);

                for(int compOffsetIter = _csr->comp_CsrOffset[nodeCompID] ; compOffsetIter < _csr->comp_CsrOffset[nodeCompID + 1] ; compOffsetIter ++){
                    int sameCompAliveNodeID = _csr->comp_NodesID_CsrData[compOffsetIter];
                    
                    if((sharedBitIndex[sameCompAliveNodeID] & bit_SI) > 0){
                        neighbor_dist_ans[sameCompAliveNodeID] = dist_arr[sameCompAliveNodeID] - 1;

                        printf("\t[1]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", sameCompAliveNodeID, neighbor_dist_ans[sameCompAliveNodeID], sameCompAliveNodeID, sharedBitIndex[sameCompAliveNodeID]);
                    }
                    else{
                        neighbor_dist_ans[sameCompAliveNodeID] = dist_arr[sameCompAliveNodeID] + 1;

                        printf("\t[2]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", sameCompAliveNodeID, neighbor_dist_ans[sameCompAliveNodeID], sameCompAliveNodeID, sharedBitIndex[sameCompAliveNodeID]);

                        if((relation[sameCompAliveNodeID] & bit_SI) > 0){
                            neighbor_dist_ans[sameCompAliveNodeID] --;
                            printf("\t\t[3]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", sameCompAliveNodeID, neighbor_dist_ans[sameCompAliveNodeID], sameCompAliveNodeID, sharedBitIndex[sameCompAliveNodeID]);
                        }
                    }

                    _CCs[sourceNeighborID] += _csr->ff[sameCompAliveNodeID] + neighbor_dist_ans[sameCompAliveNodeID] * _csr->representNode[sameCompAliveNodeID];
                    
                }
                printf("CC[%d] = %d\n", sourceNeighborID, _CCs[sourceNeighborID]);
            }

            #pragma endregion neighborOfSource_GetDist_and_AccumulationCC

        }

    }
    printf("[Check point 2]\n");
}

#pragma endregion //Function_computing


#pragma region Function_CheckingAns

void CC_CheckAns(struct CSR* _csr, int* TrueCC_Ans, int* newCC_Ans){
    for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
        if(TrueCC_Ans[nodeID] != newCC_Ans[nodeID]){
            printf("[ERROR] CC ans : TrueCC_Ans[%2d] = %2d, newCC_Ans[%2d] = %2d\n", nodeID, TrueCC_Ans[nodeID], nodeID, newCC_Ans[nodeID]);
            exit(1);
        }
    }
}

void CC_CheckDistAns(struct CSR* _csr, int* _CCs, int _tempSourceID, int* dist){
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

/**
 * @brief only compare the nodes in component
*/
void checkDistAns_exceptD1Node(struct CSR* _csr, int* newDist_Ans, int _sourceID){
    //Init dist array
    int* TrueDist_Ans = (int*)malloc(sizeof(int) * _csr->csrVSize);
    memset(TrueDist_Ans, -1, sizeof(int) * _csr->csrVSize);

    //Init Queue
    struct qQueue* Q = InitqQueue();
    qInitResize(Q, _csr->csrVSize);

    //Assign tempSourceID
    printf("[Ans Checking] SourceID = %2d ... ", _sourceID);

    //Init Traverse
    qPushBack(Q, _sourceID); 
    TrueDist_Ans[_sourceID] = 0;

    //Traverse to get distance from tempSourceID to each node in the component
    int currentNodeID   = -1;
    int neighborNodeID  = -1;
    while(!qIsEmpty(Q)){
        int currentNodeID = qPopFront(Q);
        
        for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
            neighborNodeID = _csr->csrE[neighborIndex];

            if(TrueDist_Ans[neighborNodeID] == -1){
                qPushBack(Q, neighborNodeID);
                TrueDist_Ans[neighborNodeID] = TrueDist_Ans[currentNodeID] + 1;
                // printf("TrueDist_Ans[%2d] = %2d\n", neighborNodeID, TrueDist_Ans[neighborNodeID]);
            }
        }
    }

    // //recover the dist of d1Nodes
    // int d1NodeID    = -1;
    // int d1ParentID  = -1;
    // for(int d1NodeIndex = _csr->degreeOneNodesQ->rear ; d1NodeIndex >= 0 ; d1NodeIndex --){
    //     d1NodeID    = _csr->degreeOneNodesQ->dataArr[d1NodeIndex];
    //     d1ParentID  = _csr->csrE[_csr->csrV[d1NodeID]];
        
    //     TrueDist_Ans[d1NodeID] = TrueDist_Ans[d1ParentID] + 1;
    // }

    
    // printf("\n");
    // for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
    //     printf("TrueDist_Ans[%2d] = %2d\n", nodeID, TrueDist_Ans[nodeID]);
    // }

    //compare two ans
    for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
        if(TrueDist_Ans[nodeID] != newDist_Ans[nodeID]){
            printf("[ERROR] newDist_Ans[%2d] = %2d, TrueDist_Ans[%2d] = %2d\n", nodeID, newDist_Ans[nodeID], nodeID, TrueDist_Ans[nodeID]);
            exit(1);
        }
    }

    printf("Correct !!!!\n");

    free(TrueDist_Ans);
    free(Q->dataArr);
    free(Q);
}

/**
 * @brief 
 * This check CC function can be used in "computeCC" and "computeCC_shareBased" only.
 * 
 * According to the _sourceID, traverse the graph with that _sourceID and update the TrueCC_Ans,
 * then compare "TrueCC_Ans" and "newCC_Ans" to check if there is any node that is wrong.
 * If it is, print it on the terminal.
*/
void dynamic_CC_trace_Ans(struct CSR* _csr, int* _newCC_Ans, int _sourceID){
    printf("[Ans Checking] SourceID = %6d ...", _sourceID);

    int* dist_arr = (int*)malloc(sizeof(int) * _csr->csrVSize);
    memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);

    struct qQueue* Q = InitqQueue();
    qInitResize(Q, _csr->csrVSize);
    
    dist_arr[_sourceID] = 0;
    qPushBack(Q, _sourceID);

    int currentNodeID   = -1;
    int neighborNodeID  = -1;
    int neighborIndex   = -1;
    while(!qIsEmpty(Q)){
        currentNodeID = qPopFront(Q);
        
        for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
            neighborNodeID = _csr->csrE[neighborIndex];

            if(dist_arr[neighborNodeID] == -1){
                qPushBack(Q, neighborNodeID);
                dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;
            }
        }
    }

    for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
        TrueCC_Ans[nodeID] += dist_arr[nodeID];
        if(TrueCC_Ans[nodeID] != _newCC_Ans[nodeID]){
            printf("[ERROR] while updating CC... TrueCC_Ans[%2d] = %2d, _newCC_Ans[%2d] = %2d\n", nodeID, TrueCC_Ans[nodeID], nodeID, _newCC_Ans[nodeID]);
            exit(1);
        }
    }

    CheckedNodeCount ++;
    printf("Correct !!!!Checked_Node_Count = %2d\r", CheckedNodeCount);
    free(Q->dataArr);
    free(Q);
    free(dist_arr);
    return;
}


/**
 * @brief
 * This check CC function can be used in "compute_D1_CC" and "compute_D1_CC_shareBased"
 * while they perform "notD1Node traversal" only.
*/
void dynamic_D1_CC_trace_component_Ans(struct CSR* _csr, int* _newCC_Ans, int _sourceID){
    int* dist_arr       = (int*)malloc(sizeof(int) * _csr->csrVSize);
    memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);

    struct qQueue* Q    = InitqQueue();
    qInitResize(Q, _csr->csrVSize);

    PCTL_goto_previous_line();
    PCTL_clear_line(); 
    printf("[Ans Checking] SourceID %2d ...\n", _sourceID);

    dist_arr[_sourceID] = 0;
    qPushBack(Q, _sourceID);

    int currentNodeID   = -1;
    int neighborNodeID  = -1;
    while(!qIsEmpty(Q)){
        currentNodeID = qPopFront(Q);
        
        for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
            neighborNodeID = _csr->csrE[neighborIndex];

            if(dist_arr[neighborNodeID] == -1){
                qPushBack(Q, neighborNodeID);
                dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;

                TrueCC_Ans[neighborNodeID] += _csr->ff[_sourceID] + dist_arr[neighborNodeID] * _csr->representNode[_sourceID];
            }
        }
    }
    TrueCC_Ans[_sourceID] += _csr->ff[_sourceID];
    
    int notD1NodeID = -1;
    for(int notD1NodeIndex = 0 ; notD1NodeIndex < _csr->ordinaryNodeCount ; notD1NodeIndex ++){
        notD1NodeID = _csr->notD1Node[notD1NodeID];
        if(TrueCC_Ans[notD1NodeID] != _newCC_Ans[notD1NodeID]){
            printf("[ERROR] while updating CC... TrueCC_Ans[%8d] = %8d, _newCC_Ans[%8d] = %8d\n", notD1NodeID, TrueCC_Ans[notD1NodeID], notD1NodeID, _newCC_Ans[notD1NodeID]);
            exit(1);
        }
    }

    CheckedNodeCount ++;
    PCTL_clear_line();
    printf("Correct!!!! Checked_Node_Count = %8d", CheckedNodeCount);

    free(Q->dataArr);
    free(Q);
    free(dist_arr);

    return;
}

void dynamic_D1_CC_trace_D1_Ans(struct CSR* _csr, int* _newCC_Ans, int _d1NodeID){
    int* dist_arr = (int*)malloc(sizeof(int) * _csr->csrVSize);
    memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);

    struct qQueue* Q = InitqQueue();
    qInitResize(Q, _csr->csrVSize);

    PCTL_goto_previous_line();
    PCTL_clear_line(); 
    printf("[Ans Checking] SourceID %2d ...\n", _d1NodeID);

    dist_arr[_d1NodeID] = 0;
    qPushBack(Q, _d1NodeID);

    int currentNodeID   = -1;
    int neighborNodeID  = -1;
    while(!qIsEmpty(Q)){
        currentNodeID = qPopFront(Q);

        for(int neighborIndex = _csr->oriCsrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
            neighborNodeID = _csr->csrE[neighborIndex];
            
            if(dist_arr[neighborNodeID] == -1){
                qPushBack(Q, neighborNodeID);
                dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;

                TrueCC_Ans[_d1NodeID] += dist_arr[neighborNodeID];
            }
        }
    }

    if(TrueCC_Ans[_d1NodeID] != _newCC_Ans[_d1NodeID]){
        printf("[ERROR] while updating CC ... TrueCC_Ans[%2d] = %2d, _newCC_Ans[%2d] = %2d\n", _d1NodeID, TrueCC_Ans[_d1NodeID], _d1NodeID, _newCC_Ans[_d1NodeID]);
        exit(1);
    }

    CheckedNodeCount ++;
    PCTL_clear_line();
    printf("Correct!!!! Checked_Node_Count = %8d", CheckedNodeCount);

    free(Q->dataArr);
    free(Q);
    free(dist_arr);

    return;
}

#pragma endregion //Functio_CheckingAns

int main(int argc, char* argv[]){
    char* datasetPath = argv[1];
    printf("exeName = %s\n", argv[0]);
    printf("datasetPath = %s\n", datasetPath);
    struct Graph* graph = buildGraph(datasetPath);
    struct CSR* csr     = createCSR(graph);
    
    double CC_shareBasedTime                = 0;
    double CC_ori                           = 0;
    double D1FoldingTime                    = 0;
    double D1_CC_ori                        = 0;
    double D1_CC_shareBasedTime             = 0;
    double D1_sort_CC_shareBasedTime        = 0;
    double D1_sort_CC_64Bit_shareBasedTime  = 0;
    double D1_AP_CC_ori                     = 0;
    double D1_AP_CC_shareBasedTime          = 0;

    double time1                            = 0;
    double time2                            = 0;

    int* CCs        = (int*)calloc(sizeof(int), csr->csrVSize);
    TrueCC_Ans      = (int*)calloc(sizeof(int), csr->csrVSize);
    
    #pragma region Dev
    time1 = seconds();
    compute_D1_AP_CC_shareBased_DegreeOrder(csr, csr->CCs);
    time2 = seconds();
    D1_AP_CC_shareBasedTime = time2 - time1;
    #pragma endregion //Dev



    #pragma region Release
    /************************************************************
     *                      D1Folding                           *
    ************************************************************/
    // time1 = seconds();
    // D1Folding(csr);
    // time2 = seconds();
    // D1FoldingTime = time2 - time1;
    // printf("[Execution Time] D1Folding_Time = %f\n", D1FoldingTime);


    /************************************************************
     *                      computeCC                           *
    ************************************************************/
    // time1 = seconds();
    // computeCC(csr, CCs);
    // time2 = seconds();
    // CC_ori = time2 - time1;
    // printf("[Execution Time] CC_ori = %f\n", CC_ori);
    
    
    /************************************************************
     *                  computeCC_shareBased                    *
    ************************************************************/
    // time1 = seconds();
    // computeCC_shareBased(csr, CCs);
    // time2 = seconds();
    // CC_shareBasedTime = time2 - time1;
    // printf("[Execution Time] CC_shareBased = %f\n", CC_shareBasedTime);
    

    /************************************************************
     *                      compute_D1_CC                       *
    ************************************************************/
    // time1 = seconds();
    // compute_D1_CC(csr, CCs);
    // time2 = seconds();
    // D1_CC_ori = time2 - time1;
    // printf("[Execution Time] D1_CC_ori = %f\n", D1_CC_ori);


    /************************************************************
     *                  compute_D1_CC_shareBased                *
    ************************************************************/
    // time1 = seconds();
    // compute_D1_CC_shareBased(csr, CCs);    
    // time2 = seconds();
    // D1_CC_shareBasedTime = time2 - time1;
    // printf("[Execution Time] D1_CC_shareBasedTime = %f\n", D1_CC_shareBasedTime);


    /************************************************************
     *              compute_D1_CC_shareBased_DegreeOrder        *
    ************************************************************/
    // time1 = seconds();
    // compute_D1_CC_shareBased_DegreeOrder(csr, CCs);
    // time2 = seconds();
    // D1_sort_CC_shareBasedTime = time2 - time1;
    // printf("[Execution Time] D1_sort_CC_shareBasedTime = %f\n", D1_sort_CC_shareBasedTime);


    /************************************************************
     *        compute_D1_CC_shareBased_DegreeOrder_64bit        *
    ************************************************************/
    // time1 = seconds();
    // compute_D1_CC_sharedBased_DegreeOrder_64bit(csr, CCs);
    // time2 = seconds();
    // D1_sort_CC_64Bit_shareBasedTime = time2 - time1;
    // printf("[Execution Time] D1_sort_CC_64Bit_shareBasedTime = %f\n", D1_sort_CC_64Bit_shareBasedTime);

    /************************************************************
     *                      compute_D1_AP_CC                    *
    ************************************************************/
    // time1 = seconds();
    // compute_D1_AP_CC(csr, csr->CCs);
    // time2 = seconds();
    // D1_AP_CC_ori = time2 - time1;
    // printf("[Execution Time] D1_AP_CC_ori = %f\n", D1_AP_CC_ori);
    #pragma endregion //Release

}
