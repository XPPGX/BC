/**
 * @todo 可以用普通的arr當queue，且size = csr->csrVSize
 * @todo d1Folding完成，接下來要寫個新function執行BC computation with d1Folding
*/

#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#include "headers.h"
#include <stdarg.h>
// #define DEBUG
// #define Timing
int offset = 0;
int nextBFS = 0;

int* dist_arr_next_ori;
int* dist_arr_next_method1;
int* dist_arr_method1;

float* numberOfSP_arr_next_ori;
float* numberOfSP_arr_next_method1;
float* numberOfSP_arr_method1;

struct stack* S_next_method1;
struct stack* S_method1;
struct stack* S_ori;


float* BCs_ori;

struct stack{
    int top;
    int* nodeIDs;
    // struct vVector** levelRecords;
    // int maxLevel;
    int noSharedStartIndex;
    int S_curDistStartIndex;
    int size;
    struct vVector** predecessors;
};


struct stack* initStack(int _elementSize){
    struct stack* S = (struct stack*)malloc(sizeof(struct stack));
    S->top          = -1;
    S->size         = _elementSize;
    S->nodeIDs      = (int*)calloc(sizeof(int), _elementSize);
    S->predecessors = (struct vVector**)malloc(sizeof(struct vVector*) * _elementSize);
    
    for(int i = 0 ; i < _elementSize ; i ++){
        S->predecessors[i] = InitvVector();
    }
    return S;
}

inline void stackPushNode(struct stack* _S, int _nodeID){
    _S->top ++;
    _S->nodeIDs[_S->top] = _nodeID;
}

inline void stackPushNode_shared(struct stack* _S, int _nodeID, int* _dist_arr_next){
    _S->top ++;
    _S->nodeIDs[_S->top] = _nodeID;
    if(_dist_arr_next[_S->nodeIDs[_S->top]] == _dist_arr_next[_S->nodeIDs[_S->top - 1]]){
        #ifdef DEBUG
        printf("S_next[%d] = %d, dist[%d] = %d\n", _S->top, _S->nodeIDs[_S->top], _S->nodeIDs[_S->top], _dist_arr_next[_S->nodeIDs[_S->top]]);
        #endif
        return;
    } 
    
    if(_dist_arr_next[_S->nodeIDs[_S->top]] < _dist_arr_next[_S->nodeIDs[_S->top - 1]]){
        #ifdef DEBUG
        printf("\tS_next[%d] = %d(%d), S_next[%d] = %d(%d), curDistIndex = %d, SWAP", _S->top, _S->nodeIDs[_S->top], _dist_arr_next[_S->nodeIDs[_S->top]], _S->top - 1, _S->nodeIDs[_S->top - 1], _dist_arr_next[_S->nodeIDs[_S->top - 1]], _S->S_curDistStartIndex);
        printf("(%d)<->(%d)\n", _S->nodeIDs[_S->top], _S->nodeIDs[_S->S_curDistStartIndex]);
        #endif
        swap(&(_S->nodeIDs[_S->top]), &(_S->nodeIDs[_S->S_curDistStartIndex]));
        _S->S_curDistStartIndex ++;
        return;
    }
    else if(_dist_arr_next[_S->nodeIDs[_S->top]] > _dist_arr_next[_S->nodeIDs[_S->top - 1]]){
        // _S->S_prevDistStartIndex = _S->S_curDistStartIndex;
        _S->S_curDistStartIndex = _S->top;
        #ifdef DEBUG
        printf("\tS_next.curIndex = %d\n", _S->S_curDistStartIndex);
        #endif
    }
}

inline void stackPushNeighbor(struct stack* _S, int _nodeID, int _predecessor){
    vAppend(_S->predecessors[_nodeID], _predecessor);
}

struct vVector** InitLevelRecords(int _elementSize){
    struct vVector** LevelRecords = (struct vVector**)malloc(sizeof(struct vVector*) * _elementSize);
    for(int i = 0 ; i < _elementSize ; i ++){
        LevelRecords[i] = InitvVector();
    }
    return LevelRecords;
}

// inline void pushLevelNode(struct stack* _S, int _nodeID, int _dist){
//     if(_S->maxLevel < _dist)
//         _S->maxLevel = _dist;
//     vAppend(_S->levelRecords[_dist], _nodeID);

//     #ifndef Timing
//     #ifdef DEBUG
//     printf("push %d to nextBFS level %d\n", _nodeID, _dist);
//     #endif
//     #endif
// }

// void resetLevelRecords(struct stack* _S){
//     for(int i = 0 ; i <= _S->maxLevel ; i ++){
//         _S->levelRecords[i]->tail = -1;
//     }
//     _S->maxLevel = 0;
// }

/**
 * @brief 返回Stack Top node的vector of predecessors
 * @param _S                a pointer of stack
 * @param _predecessors     point to the predecessors of the stackTopNodeID
 * @param _stackTopNodeID   point to the address which stores the stackTopNodeID
 * @return                  a pointer of vector which is the predecessors of the top node in stack _S
*/
void stackPop(struct stack* _S, struct vVector** _predecessors, int* _stackTopNodeID){
    *_stackTopNodeID    = _S->nodeIDs[_S->top];
    *_predecessors      = _S->predecessors[*_stackTopNodeID];

    _S->top --;
}

int stackIsEmpty(struct stack* _S){
    if(_S->top != -1){
        return 0;
    }
    return 1;
}

inline void resetQueue(struct qQueue* _Q){
    _Q->front = 0;
    _Q->rear = -1;
    //Q->size如果不變，就不須memcpy
    // _Q->size = 5;
}

void resetStack(struct stack* _S){
    _S->top = -1;
    for(int i = 0 ; i < _S->size ; i++){
        _S->predecessors[i]->tail = -1;
    }
}

inline void resizeVec(struct vVector* _vec, int _size){
    free(_vec->dataArr);
    _vec->dataArr   = (int*)malloc(sizeof(int) * _size);
    _vec->size      = _size;
}

inline void swapVec(struct vVector** _tempVecs, int _distChange){
    _tempVecs[2]->tail = -1;
    struct vVector* tempVecPtr = _tempVecs[2];

    // _tempVecs[2] = _tempVecs[(2 - _distChange) * (_distChange <= 2) + (2 * (_distChange > 2))];
    
    // _tempVecs[1] = _tempVecs[1 - (_distChange == 1)];
    // _tempVecs[1]->tail = _tempVecs[1]->tail * (_distChange == 1) + (-1 * (_distChange > 1));

    // _tempVecs[0] = (_distChange > 2) ? _tempVecs[0] : tempVecPtr;
    // _tempVecs[0]->tail = _tempVecs[0]->tail * (_distChange <= 2) + (-1 * (_distChange > 2));

#pragma region originBranch
    if(_distChange == 1){
        _tempVecs[2] = _tempVecs[1];
        _tempVecs[1] = _tempVecs[0];
        _tempVecs[0] = tempVecPtr;
    }
    else if(_distChange == 2){
        _tempVecs[2] = _tempVecs[0];
        _tempVecs[1]->tail = -1;
        _tempVecs[0] = tempVecPtr;

    }
    else{
        _tempVecs[1]->tail = -1;
        _tempVecs[0]->tail = -1;
    }
#pragma endregion //originBranch
}

void bit32(void *v){
    unsigned mask=0x80000000U;
    unsigned value = *(unsigned*)v;
    while(mask){
        printf("%d", (mask&value)!=0U);
        mask>>=1;
    }
}

void computeBC(struct CSR* _csr, float* _BCs){

    float* numberOfSP_arr   = (float*)calloc(sizeof(float), _csr->csrVSize);
    float* dependencies_arr = (float*)calloc(sizeof(float), _csr->csrVSize);
    int* dist_arr           = (int*)calloc(sizeof(int), _csr->csrVSize);
    //宣告Queue，用於forward traverse
    struct qQueue* Q    = InitqQueue();
    qInitResize(Q, _csr->csrVSize);
    //宣告stack，用於backward traverse
    struct stack* S     = initStack(_csr->csrVSize);

    #ifdef DEBUG
    printf("startNodeID = %d, endNodeID = %d\n", _csr->startNodeID, _csr->endNodeID);
    #endif
    #ifdef Timing
    double time1 = seconds();
    #endif
    for(int sourceID = nextBFS ; sourceID <= _csr->endNodeID ; sourceID ++){
        memset(numberOfSP_arr, 0, sizeof(float) * _csr->csrVSize);
        memset(dependencies_arr, 0, sizeof(float) * _csr->csrVSize);
        memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);
        resetQueue(Q);
        resetStack(S);
        int sourceNodeID                = sourceID;    //SourceNodeID
        // printf("sourceID = %d\n", sourceNodeID);
        numberOfSP_arr[sourceNodeID]    = 1;
        dist_arr[sourceNodeID]          = 0;
        // //宣告Queue，用於forward traverse
        // struct qQueue* Q    = InitqQueue();
        // //宣告stack，用於backward traverse
        // struct stack* S     = initStack(_csr->csrVSize);
        qPushBack(Q, sourceNodeID);
        int currentNodeID   = -1;
        int neighborNodeID  = -1;
        int count = 0;
        while(!qIsEmpty(Q)){
            currentNodeID = qPopFront(Q);

            #ifdef DEBUG
            printf("current[%2d]\n", currentNodeID);
            #endif

            stackPushNode(S, currentNodeID);
            for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
                neighborNodeID = _csr->csrE[neighborIndex];
                if(dist_arr[neighborNodeID] < 0){
                    qPushBack(Q, neighborNodeID);
                    dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;

                    #ifdef DEBUG
                    printf("\tnextLevel.append(%2d), dist = %2d\n", neighborNodeID, dist_arr[neighborNodeID]);
                    #endif
                }
                
                if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){
                    numberOfSP_arr[neighborNodeID] += numberOfSP_arr[currentNodeID];
                    stackPushNeighbor(S, neighborNodeID, currentNodeID);
                }
            }
        }

        // int testNodeID = 228;
        // printf("dist[%d] = %d, ori_pred[%d] = {", testNodeID, dist_arr[testNodeID], testNodeID);
        // for(int i = 0 ; i <= S->predecessors[testNodeID]->tail ; i ++){
        //     printf("%d(%d), ", S->predecessors[testNodeID]->dataArr[i], dist_arr[S->predecessors[testNodeID]->dataArr[i]]);
        // }
        // printf("}\n");
        // printf("dist[11566] = %d\n", dist_arr[11566]);
        

        #ifdef DEBUG
        printf("\n\nSTACK\n");
        for(int Iter = S->top ; Iter != -1 ; Iter--){
            printf("S[%2d].NodeID = %2d, Predecessors = {", Iter, S->nodeIDs[Iter]);

            for(int predIndex = 0 ; predIndex <= S->predecessors[S->nodeIDs[Iter]]->tail ; predIndex ++){
                printf("%2d, ", S->predecessors[S->nodeIDs[Iter]]->dataArr[predIndex]);
            }
            printf("}\n");
        }
        #endif

        memset(dependencies_arr, 0, sizeof(int) * _csr->csrVSize);
        struct vVector* predecessors    = NULL;
        int stackTopNodeID              = -1;
        int predecessorIndex            = -1;
        int predecessorID               = -1;
        while(!stackIsEmpty(S)){
            stackPop(S, &predecessors, &stackTopNodeID);
            for(predecessorIndex = 0 ; predecessorIndex <= predecessors->tail ; predecessorIndex ++){
                predecessorID = predecessors->dataArr[predecessorIndex];
                
                dependencies_arr[predecessorID] += ((numberOfSP_arr[predecessorID] / numberOfSP_arr[stackTopNodeID]) * (1 + dependencies_arr[stackTopNodeID]));
                if(stackTopNodeID != sourceNodeID){
                    _BCs[stackTopNodeID]       += dependencies_arr[stackTopNodeID]; 
                }
            }
        }
        #ifdef Timing
        double time2 = seconds();
        printf("[Execution Time] BC_oriForward = %2f\n", time2-time1);
        #endif
        // for(int nodeIDIter = _csr->startNodeID ; nodeIDIter <= _csr->endNodeID ; nodeIDIter ++){
        //     printf("BC[%2d] = %2f\n", nodeIDIter, _BCs[nodeIDIter]);
        // }
        break;
    }
    #ifdef DEBUG
    for(int nodeIDIter = _csr->startNodeID ; nodeIDIter <= _csr->endNodeID ; nodeIDIter ++){
        printf("BC[%2d] = %2f\n", nodeIDIter, _BCs[nodeIDIter]);
    }
    #endif

    // free(numberOfSP_arr);
    numberOfSP_arr_next_ori = numberOfSP_arr;
    free(dependencies_arr);
    // free(dist_arr);
    dist_arr_next_ori = dist_arr;

    free(Q->dataArr);
    free(Q);
    // for(int i = 0 ; i < _csr->csrVSize ; i ++){
    //     free(S->predecessors[i]->dataArr);
    //     free(S->predecessors[i]);
    // }
    // free(S->nodeIDs);
    // free(S->predecessors);
    S_ori = S;
}

void computeBC_shareBased(struct CSR* _csr, float* _BCs){
    // showCSR(_csr);

    //先排序nodesDegree數，然後從degree小的開始當first BFS source然後再找second BFS source
    float* numberOfSP_arr   = (float*)calloc(sizeof(float), _csr->csrVSize);
    float* dependencies_arr = (float*)calloc(sizeof(float), _csr->csrVSize);
    int* dist_arr           = (int*)calloc(sizeof(int), _csr->csrVSize);
    struct qQueue* Q        = InitqQueue();
    qInitResize(Q, _csr->csrVSize);
    struct stack* S         = initStack(_csr->csrVSize);

    float* numberOfSP_arr_next      = (float*)calloc(sizeof(float), _csr->csrVSize);
    float* dependencies_arr_next    = (float*)calloc(sizeof(float), _csr->csrVSize);
    int* dist_arr_next              = (int*)calloc(sizeof(int), _csr->csrVSize);
    struct stack* S_next            = initStack(_csr->csrVSize);
    int* relations_next             = (int*)calloc(sizeof(int), _csr->csrVSize);
    int* nodesDone                  = (int*)calloc(sizeof(int), _csr->csrVSize);
    struct vVector **tempVecs = (struct vVector**)malloc(sizeof(struct vVector*) * 3);
    tempVecs[0] = InitvVector();
    tempVecs[1] = InitvVector();
    tempVecs[2] = InitvVector();
    resizeVec(tempVecs[0], _csr->csrVSize);
    resizeVec(tempVecs[1], _csr->csrVSize);
    resizeVec(tempVecs[2], _csr->csrVSize);
    // S_next->levelRecords            = InitLevelRecords(_csr->csrVSize);

    #ifdef DEBUG
    printf("startNodeID = %d, endNodeID = %d\n", _csr->startNodeID, _csr->endNodeID);
    #endif
    
    int minDegreeNeighborID;
    int minDegree;
    int neighborID;
    double time1 = seconds();
    for(int sourceID = offset ; sourceID <= _csr->endNodeID ; sourceID ++){
        if(nodesDone[sourceID] == 1){
            printf("sourceID = %d done.\n", sourceID);
            continue;
        }
        printf("sourceID = %d start\t", sourceID);
        nodesDone[sourceID] = 1;
        
        //找自己的鄰居中，nodesDone != 1且degree最少的。
        minDegreeNeighborID = -1;
        minDegree           = __INT_MAX__;
        neighborID          = -1;
        //找Degree小的當下一個source
        for(int neighborIndex = _csr->csrV[sourceID] ; neighborIndex < _csr->csrV[sourceID + 1] ; neighborIndex ++){
            neighborID              = _csr->csrE[neighborIndex];

            if(nodesDone[neighborID] == 1){continue;}
            //多加"_csr->csrNodesDegree[neighborID] != 1"，因為還沒寫d1 folding
            // if(_csr->csrNodesDegree[neighborID] != 1){
            //     minDegreeNeighborID = neighborID;
            //     break;
            // }
            if(minDegree > _csr->csrNodesDegree[neighborID] && _csr->csrNodesDegree[neighborID] != 1){
                // minDegree           = _csr->csrNodesDegree[neighborID];
                minDegreeNeighborID = neighborID;
                break;
            }
        }

        memset(numberOfSP_arr, 0, sizeof(float) * _csr->csrVSize);
        memset(dependencies_arr, 0, sizeof(float) * _csr->csrVSize);
        memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);
        memset(relations_next, 0, sizeof(int) * _csr->csrVSize);
        resetQueue(Q);
        resetStack(S);
        
        if(minDegreeNeighborID != -1){
            //For test
            nextBFS = minDegreeNeighborID;
            printf("nextBFS = %d\n", nextBFS);
            // printf("Degree[%d] = %d\n", nextBFS, _csr->csrNodesDegree[nextBFS]);
            #ifdef DEBUG
            printf("\nFind next sourceID = %d\n", minDegreeNeighborID);
            printf("Reset next source info...\n");
            #endif

            memset(numberOfSP_arr_next, 0, sizeof(float) * _csr->csrVSize);
            memset(dependencies_arr_next, 0, sizeof(float) * _csr->csrVSize);
            memset(dist_arr_next, -1, sizeof(int) * _csr->csrVSize);
            // memset(relations_next, 0, sizeof(int) * _csr->csrVSize);
            resetStack(S_next);
            
            numberOfSP_arr_next[minDegreeNeighborID]    = 1;
            // printf("numberOfSP_next[%d] = %f\n", minDegreeNeighborID, numberOfSP_arr_next[minDegreeNeighborID]);
            dist_arr_next[minDegreeNeighborID]          = 0;
            relations_next[minDegreeNeighborID]         = 1;
            nodesDone[minDegreeNeighborID]              = 1;
        }
        else{
            printf("\n");
        }
        numberOfSP_arr[sourceID]    = 1;
        dist_arr[sourceID]          = 0;
        
        qPushBack(Q, sourceID);
        register int currentNodeID           = -1;
        register int neighborNodeID          = -1;
        //forward traverse with info sharing
        while(!qIsEmpty(Q)){
            currentNodeID = qPopFront(Q);
            stackPushNode(S, currentNodeID);
            #ifdef DEBUG
            printf("currentNodeID = %d, ", currentNodeID);
            #endif

            if(relations_next[currentNodeID] == 0){
                #ifdef DEBUG
                printf("relations[%d] = 0\n", currentNodeID);
                #endif

                for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborNodeID = _csr->csrE[neighborIndex];
                    if(dist_arr[neighborNodeID] < 0){
                        qPushBack(Q, neighborNodeID);
                        dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;
                    }
                    if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){
                        numberOfSP_arr[neighborNodeID] += numberOfSP_arr[currentNodeID];
                        stackPushNeighbor(S, neighborNodeID, currentNodeID);
                    }
                }
            }
            else if(relations_next[currentNodeID] == 1){
                #ifdef DEBUG
                printf("relations[%d] = 1, {", currentNodeID);
                #endif

                stackPushNode(S_next, currentNodeID);
                
                for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborNodeID = _csr->csrE[neighborIndex];
                    if(dist_arr[neighborNodeID] < 0){
                        qPushBack(Q, neighborNodeID);
                        dist_arr[neighborNodeID]        = dist_arr[currentNodeID] + 1;
                        dist_arr_next[neighborNodeID]   = dist_arr_next[currentNodeID] + 1;

                    }
                    if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){
                        numberOfSP_arr[neighborNodeID]      += numberOfSP_arr[currentNodeID];
                        numberOfSP_arr_next[neighborNodeID] += numberOfSP_arr_next[currentNodeID];
                        
                        dist_arr_next[neighborNodeID]   = dist_arr_next[currentNodeID] + 1;
                        relations_next[neighborNodeID]  = 1;

                        #ifdef DEBUG
                        printf("%d, ", neighborNodeID);
                        #endif

                        stackPushNeighbor(S, neighborNodeID, currentNodeID);
                        stackPushNeighbor(S_next, neighborNodeID, currentNodeID);
                    }
                }
                #ifdef DEBUG
                printf("}\n");
                printf("S_next[%d] = %d,\tdist[%d] = %d\n", S_next->top, S_next->nodeIDs[S_next->top], S_next->nodeIDs[S_next->top], dist_arr_next[S_next->nodeIDs[S_next->top]]);
                #endif
            }
        }
        if(minDegreeNeighborID != -1){
            resetQueue(Q);
            currentNodeID = minDegreeNeighborID;
            // printf("numberofSP[%d] = %f\n", currentNodeID, numberOfSP_arr_next[currentNodeID]);
            int minDist = __INT_MAX__;
            int predecessorID = -1;
            int successorID = -1;
            
            // int predVecIndex = 0;
            // int sameLevelVecIndex = 1;
            // int succVecIndex = 2;
            int distChange = -1;
            #ifdef DEBUG
            printf("Q : {");
            #endif

            for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
                neighborID = _csr->csrE[neighborIndex];
                if(relations_next[neighborID] == 0){
                    #ifdef DEBUG
                    printf("none visited %d\n", neighborID);
                    #endif

                    qPushBack(Q, neighborID);
                    relations_next[neighborID] = 2;
                }
            }

            S_next->S_curDistStartIndex = S_next->top + 1;
            S_next->noSharedStartIndex  = S_next->top + 1;
            // printf("\t NEW : noSharedStartIndex = %d\n", S_next->noSharedStartIndex);
            while(!qIsEmpty(Q)){
                minDist = __INT_MAX__;
                currentNodeID = qPopFront(Q);
                
                #ifdef DEBUG
                printf("%d : \n", currentNodeID);
                #endif
                tempVecs[0]->tail   = -1;
                tempVecs[1]->tail   = -1;
                tempVecs[2]->tail   = -1;
                distChange          = -1;
                for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborID = _csr->csrE[neighborIndex];
                    
                    if(relations_next[neighborID] == 0){
                        qPushBack(Q, neighborID);

                        #ifdef DEBUG
                        printf("\tpush to Q %d\n", neighborID);
                        #endif
                        //relations are for DEBUG
                        // relations_next[neighborID] = relations_next[currentNodeID] + 1;
                        relations_next[neighborID] = 2;
                    }
                    else{
                        #ifdef DEBUG
                        printf("\tdist_arr_next[%d] = %d\n", neighborID, dist_arr_next[neighborID]);
                        #endif

                        if(dist_arr_next[neighborID] > -1){
                            if(minDist > dist_arr_next[neighborID]){
                                //update distChange
                                distChange = minDist - dist_arr_next[neighborID];
                                //距離變化不會超過2
                                // if(minDist != __INT_MAX__ && distChange > 2){
                                //     printf("currentNodeID = %d, distChagne = %d\n", currentNodeID, distChange);
                                // }
                                //update minimum distance
                                minDist = dist_arr_next[neighborID];

                                //update the dist of currentNodeID
                                dist_arr_next[currentNodeID] = dist_arr_next[neighborID] + 1;

                                #ifdef DEBUG
                                printf("\tminDist -> %d, dist[%d] = %d\n", minDist, currentNodeID, dist_arr_next[currentNodeID]);
                                #endif

                                //exchange the level of vectors, let predVec have clean vector
                                swapVec(tempVecs, distChange);
                                
                                //reset number of shortest path
                                numberOfSP_arr_next[currentNodeID] = 0;
                            }

                            //push neighbors into correct vectors
                            if(dist_arr_next[neighborID] + 1 == dist_arr_next[currentNodeID]){
                                vAppend(tempVecs[0], neighborID);

                                #ifdef DEBUG
                                printf("\t[0]push %d (predecessor), in predVec[%d]\n", neighborID, tempVecs[0]->tail);
                                #endif
                            }
                            else if(dist_arr_next[neighborID] == dist_arr_next[currentNodeID]){
                                vAppend(tempVecs[1], neighborID); 

                                #ifdef DEBUG
                                printf("\t[1] push %d (same level), sameLevelVec[%d]\n", neighborID, tempVecs[1]->tail);
                                #endif
                            }
                            else if(dist_arr_next[neighborID] == dist_arr_next[currentNodeID] + 1){
                                vAppend(tempVecs[2], neighborID);

                                #ifdef DEBUG
                                printf("\t[2] push %d (successor), in succVec[%d]\n", neighborID, tempVecs[2]->tail);
                                #endif
                            }
                        }
                    }
                }

                stackPushNode_shared(S_next, currentNodeID, dist_arr_next);

                //耗時最多的部分，在memcpy
                resizeVec(S_next->predecessors[currentNodeID], (tempVecs[0]->tail + 1));
                memcpy(S_next->predecessors[currentNodeID]->dataArr, tempVecs[0]->dataArr, sizeof(int) * (tempVecs[0]->tail + 1));
                S_next->predecessors[currentNodeID]->tail = tempVecs[0]->tail;
                #ifdef DEBUG
                printf("\ttail = %d\n", S_next->predecessors[currentNodeID]->tail);
                #endif

                for(int succIndex = 0 ; succIndex <= tempVecs[2]->tail ; succIndex ++){
                    successorID = tempVecs[2]->dataArr[succIndex];
                    #ifdef DEBUG
                    printf("\tnode %d add pred %d\n", successorID, currentNodeID);
                    #endif
                    stackPushNeighbor(S_next, successorID, currentNodeID);
                }
                #ifdef DEBUG
                printf("\t# dist_arr_next[%d] = %d\n", currentNodeID, dist_arr_next[currentNodeID]);
                printf("\tS_next[%2d].NodeID = %2d, Predecessors = {", S_next->top, S_next->nodeIDs[S_next->top]);
                for(int predIndex = 0 ; predIndex <= S_next->predecessors[currentNodeID]->tail ; predIndex ++){
                    printf("%d(%d), ", predIndex, S_next->predecessors[currentNodeID]->dataArr[predIndex]);
                }
                printf("}\n");
                #endif
            }
            

            //這裡可以用stack紀錄node先後順序，把距離變化為2跟前面的node的作swap
            int nodeID = 0;
            int predID = 0;
            for(int indexIter = S_next->noSharedStartIndex ; indexIter <= S_next->top ; indexIter ++){
                nodeID = S_next->nodeIDs[indexIter];
                for(int predIndexIter = 0 ; predIndexIter <= S_next->predecessors[nodeID]->tail ; predIndexIter ++){
                    predID = S_next->predecessors[nodeID]->dataArr[predIndexIter];
                    
                    numberOfSP_arr_next[nodeID] += numberOfSP_arr_next[predID];
                    #ifdef DEBUG
                    printf("%d(%f), ", predID, numberOfSP_arr_next[predID]);
                    #endif
                }
                #ifdef DEBUG
                printf(" => numberOfSP[%d] = %f\n", nodeID, numberOfSP_arr_next[nodeID]);
                #endif
            }
        }
        

#pragma region stackSequenceChecking
        for(int i = 1 ; i <= S->top ; i ++){
            if(S->nodeIDs[i - 1] <= _csr->startNodeID){
                continue;
            }
            if(dist_arr[S->nodeIDs[i]] < dist_arr[S->nodeIDs[i - 1]]){
                printf("Stack1 Sequence wrong => S[i = %d] = %d, dist[%d] = %d, ", i, S->nodeIDs[i], S->nodeIDs[i], dist_arr[S->nodeIDs[i]]);
                printf("S[i - 1 = %d] = %d, dist[%d] = %d\n", i - 1, S->nodeIDs[i - 1], S->nodeIDs[i - 1], dist_arr[S->nodeIDs[i - 1]]);
                exit(1);
            }
        }
        if(minDegreeNeighborID != -1){
            for(int i = S_next->noSharedStartIndex ; i <= S_next->top ; i ++){
                if(i - 1 < S_next->noSharedStartIndex){
                    continue;
                }
                if(dist_arr_next[S_next->nodeIDs[i]] < dist_arr_next[S_next->nodeIDs[i - 1]]){
                    printf("Stack2 Sequence wrong => S[i = %d] = %d, dist[%d] = %d, ", i, S_next->nodeIDs[i], S_next->nodeIDs[i], dist_arr_next[S_next->nodeIDs[i]]);
                    printf("S[i - 1 = %d] = %d, dist[%d] = %d\n", i - 1, S_next->nodeIDs[i - 1], S_next->nodeIDs[i - 1], dist_arr_next[S_next->nodeIDs[i - 1]]);
                    exit(1);
                }
            }
        }
#pragma endregion
        
#pragma region backwardTraverse
        struct vVector* predecessors    = NULL;
        int stackTopNodeID              = -1;
        int predecessorIndex            = -1;
        int predecessorID               = -1;
        printf("source %d :\n", sourceID);
        while(!stackIsEmpty(S)){
            stackPop(S, &predecessors, &stackTopNodeID);
            for(predecessorIndex = 0 ; predecessorIndex <= predecessors->tail ; predecessorIndex ++){
                predecessorID = predecessors->dataArr[predecessorIndex];
                // printf("numberOfSP_arr : %2d(%f)p, %2d(%f)s, ", predecessorID, numberOfSP_arr[predecessorID], stackTopNodeID, numberOfSP_arr[stackTopNodeID]);
                dependencies_arr[predecessorID] += ((numberOfSP_arr[predecessorID] / numberOfSP_arr[stackTopNodeID]) * (1 + dependencies_arr[stackTopNodeID]));
                // printf("dependency[%2d] = %f, _BCs[%2d] = %f\n", predecessorID, dependencies_arr[predecessorID], predecessorID, _BCs[predecessorID]);
                if(stackTopNodeID != sourceID){
                    _BCs[stackTopNodeID] += dependencies_arr[stackTopNodeID];
                }
            }
        }
        if(minDegreeNeighborID != -1){
            // printf("\n\nnextBFS %d :\n", nextBFS);
            while(!stackIsEmpty(S_next)){
                stackPop(S_next, &predecessors, &stackTopNodeID);
                for(predecessorIndex = 0 ; predecessorIndex <= predecessors->tail ; predecessorIndex ++){
                    predecessorID = predecessors->dataArr[predecessorIndex];
                    // printf("numberOfSP_arr : %2d(%f)p, %2d(%f)s, ", predecessorID, numberOfSP_arr[predecessorID], stackTopNodeID, numberOfSP_arr[stackTopNodeID]);
                    dependencies_arr_next[predecessorID] += ((numberOfSP_arr_next[predecessorID] / numberOfSP_arr_next[stackTopNodeID]) * (1 + dependencies_arr_next[stackTopNodeID]));
                    // printf("dependency[%2d] = %f, _BCs[%2d] = %f\n", dependencies_arr[predecessorID], predecessorID, _BCs[predecessorID]);
                    if(stackTopNodeID != minDegreeNeighborID){
                        _BCs[stackTopNodeID] += dependencies_arr_next[stackTopNodeID];
                    }
                }
            }
        }
#pragma endregion

        #ifdef Timing
        double time2 = seconds();
        printf("[Execution Time] BC_sharedForward = %2f\n", time2 - time1);
        #endif
        
        nextBFS = sourceID;
        computeBC(_csr, BCs_ori);
        printf("sourceID = %d, checking Ans...\n", sourceID);
        check_SPandDist_Ans2(_csr, numberOfSP_arr, dist_arr);
        checkStackans2(S_ori, S, _csr);

        if(minDegreeNeighborID != -1){
            nextBFS = minDegreeNeighborID;
            computeBC(_csr, BCs_ori);
            printf("nextBFS = %d, checking Ans...\n", minDegreeNeighborID);
            check_SPandDist_Ans2(_csr, numberOfSP_arr_next, dist_arr_next);
            checkStackans2(S_ori, S_next, _csr);
        }
        // break;
    }
    
    free(tempVecs[0]->dataArr);
    free(tempVecs[1]->dataArr);
    free(tempVecs[2]->dataArr);
    free(tempVecs[0]);
    free(tempVecs[1]);
    free(tempVecs[2]);
    free(tempVecs);

    free(Q->dataArr);
    free(Q);
    free(numberOfSP_arr);
    // numberOfSP_arr_method1 = numberOfSP_arr;
    free(numberOfSP_arr_next);
    // numberOfSP_arr_next_method1 = numberOfSP_arr_next;
    free(dependencies_arr);
    free(dependencies_arr_next);
    free(dist_arr);
    // dist_arr_method1 = dist_arr;
    free(dist_arr_next);
    // dist_arr_next_method1 = dist_arr_next;
    free(relations_next);
    free(nodesDone);
    
    // S_method1 = S;
    // S_next_method1 = S_next;
    for(int i = 0 ; i < _csr->csrVSize ; i ++){
        free(S->predecessors[i]->dataArr);
        free(S->predecessors[i]);

        free(S_next->predecessors[i]->dataArr);
        free(S_next->predecessors[i]);
    }
    free(S->nodeIDs);
    free(S->predecessors);
    free(S);
    free(S_next->nodeIDs);
    free(S_next->predecessors);
    free(S_next);
}
int checkStackans2(struct stack* S_origin, struct stack* S_method1, struct CSR* _csr){
    // printf("\n\n");
    printf("Check Ans...\n");
    int* existArr = calloc(_csr->csrVSize, sizeof(int));
    for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
        memset(existArr, 0, sizeof(int) * _csr->startNodeID);
        // printf("nodeID = %d : \n", nodeID);
        if(S_origin->predecessors[nodeID]->tail != S_method1->predecessors[nodeID]->tail){
            printf("predecessor[%d] : Ori.tail = %d, method1.tail = %d => len diff\n", nodeID, S_origin->predecessors[nodeID]->tail, S_method1->predecessors[nodeID]->tail);
            exit(1);
        }
        

        // printf("\tOri_Pred \t= {");
        for(int i = 0 ; i <= S_origin->predecessors[nodeID]->tail ; i ++){
            existArr[S_origin->predecessors[nodeID]->dataArr[i]] = 1;
            // printf("%d, ", S_origin->predecessors[nodeID]->dataArr[i]);
        }
        // printf("}\n");


        // printf("\tMethod1_Pred \t= {");
        for(int i = 0 ; i <= S_method1->predecessors[nodeID]->tail ; i++){
            if(existArr[S_method1->predecessors[nodeID]->dataArr[i]] == 0){
                printf("predecessor[%d] : Ori, method1 => data diff\n", nodeID);
                exit(1);
            }
            // printf("%d, ", S_method1->predecessors[nodeID]->dataArr[i]);
        }
        // printf("}\n");
    }
    
    printf("done\n\n");

    free(existArr);
    for(int Iter = 0 ; Iter < _csr->csrVSize ; Iter ++){
        free(S_origin->predecessors[Iter]->dataArr);
        free(S_origin->predecessors[Iter]);
        // free(S_method1->predecessors[Iter]->dataArr);
        // free(S_method1->predecessors[Iter]);
    }
    free(S_origin->nodeIDs);
    // free(S_method1->nodeIDs);
    free(S_origin->predecessors);
    // free(S_method1->predecessors);
    free(S_origin);
    // free(S_method1);
    return 1;
}

int checkStackans(struct stack* S_origin, struct stack* S_method1, struct CSR* _csr){
    // printf("\n\n");
    printf("Check Ans...\n");
    int* existArr = calloc(_csr->csrVSize, sizeof(int));
    for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
        memset(existArr, 0, sizeof(int) * _csr->startNodeID);
        // printf("nodeID = %d : \n", nodeID);
        if(S_origin->predecessors[nodeID]->tail != S_method1->predecessors[nodeID]->tail){
            printf("predecessor[%d] : Ori.tail = %d, method1.tail = %d => len diff\n", nodeID, S_origin->predecessors[nodeID]->tail, S_method1->predecessors[nodeID]->tail);
            exit(1);
        }
        

        // printf("\tOri_Pred \t= {");
        for(int i = 0 ; i <= S_origin->predecessors[nodeID]->tail ; i ++){
            existArr[S_origin->predecessors[nodeID]->dataArr[i]] = 1;
            // printf("%d, ", S_origin->predecessors[nodeID]->dataArr[i]);
        }
        // printf("}\n");


        // printf("\tMethod1_Pred \t= {");
        for(int i = 0 ; i <= S_method1->predecessors[nodeID]->tail ; i++){
            if(existArr[S_method1->predecessors[nodeID]->dataArr[i]] == 0){
                printf("predecessor[%d] : Ori, method1 => data diff\n", nodeID);
                exit(1);
            }
            // printf("%d, ", S_method1->predecessors[nodeID]->dataArr[i]);
        }
        // printf("}\n");
    }
    
    printf("done\n\n");

    free(existArr);
    for(int Iter = 0 ; Iter < _csr->csrVSize ; Iter ++){
        free(S_origin->predecessors[Iter]->dataArr);
        free(S_origin->predecessors[Iter]);
        free(S_method1->predecessors[Iter]->dataArr);
        free(S_method1->predecessors[Iter]);
    }
    free(S_origin->nodeIDs);
    free(S_method1->nodeIDs);
    free(S_origin->predecessors);
    free(S_method1->predecessors);
    free(S_origin);
    free(S_method1);
    
    return 1;
}

int check_SPandDist_Ans2(struct CSR* _csr, float* numberOfSP_arr_Checking, int* dist_arr_Checking){
    
    for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID++){
        if(dist_arr_next_ori[nodeID] != dist_arr_Checking[nodeID]){
            printf("\nori_dist[%d] = %d, method1_dist[%d] = %d, dist_arr ans diff\n", nodeID, dist_arr_next_ori[nodeID], nodeID, dist_arr_Checking[nodeID]);
            exit(1);
        }

        if(numberOfSP_arr_next_ori[nodeID] != numberOfSP_arr_Checking[nodeID]){
            printf("\nori_SP[%d] = %f, method1_SP[%d] = %f, numberofSP ans diff\n", nodeID, numberOfSP_arr_next_ori[nodeID], nodeID, numberOfSP_arr_Checking[nodeID]);
            exit(1);
        }
    }
    free(dist_arr_next_ori);
    free(numberOfSP_arr_next_ori);

    // free(numberOfSP_arr_Checking);
    // free(dist_arr_Checking);

}

int check_SPandDist_Ans(struct CSR* _csr){
    for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID++){
        // if(dist_arr_next_ori[nodeID] != dist_arr_next_method1[nodeID]){
        //     printf("\nori_dist[%d] = %d, method1_dist[%d] = %d, dist_arr ans diff\n", nodeID, dist_arr_next_ori[nodeID], nodeID, dist_arr_next_method1[nodeID]);
        //     exit(1);
        // }

        // if(numberOfSP_arr_next_ori[nodeID] != numberOfSP_arr_next_method1[nodeID]){
        //     printf("\nori_SP[%d] = %f, method1_SP[%d] = %f, numberofSP ans diff\n", nodeID, numberOfSP_arr_next_ori[nodeID], nodeID, numberOfSP_arr_next_method1[nodeID]);
        //     exit(1);
        // }
        if(dist_arr_next_ori[nodeID] != dist_arr_method1[nodeID]){
            printf("\nori_dist[%d] = %d, method1_dist[%d] = %d, dist_arr ans diff\n", nodeID, dist_arr_next_ori[nodeID], nodeID, dist_arr_method1[nodeID]);
            exit(1);
        }

        if(numberOfSP_arr_next_ori[nodeID] != numberOfSP_arr_method1[nodeID]){
            printf("\nori_SP[%d] = %f, method1_SP[%d] = %f, numberofSP ans diff\n", nodeID, numberOfSP_arr_next_ori[nodeID], nodeID, numberOfSP_arr_method1[nodeID]);
            exit(1);
        }
    }
    free(dist_arr_next_ori);
    free(numberOfSP_arr_next_ori);

    free(dist_arr_method1);
    free(numberOfSP_arr_method1);
    // free(dist_arr_next_method1);
    // free(numberOfSP_arr_next_method1);
}

int main(int argc, char* argv[]){
    char* datasetPath = argv[1];
    printf("exeName = %s\n", argv[0]);
    printf("datasetPath = %s\n", datasetPath);
    struct Graph* graph = buildGraph(datasetPath);
    struct CSR* csr     = createCSR(graph);
    // showCSR(csr);
    float* BCs          = (float*)calloc(sizeof(float), csr->csrVSize);
    float* BCs2         = (float*)calloc(sizeof(float), csr->csrVSize);
    BCs_ori             = (float*)calloc(sizeof(float), csr->csrVSize);
    double time1        = 0;
    double time2        = 0;
    double BrandesTime  = 0;
    time1               = seconds();
    offset              = 1;
    computeBC_shareBased(csr, BCs);
    // struct stack* S_method1 = computeBC_shareBased(csr, BCs);;
    // struct stack* S_ori     = computeBC(csr, BCs);
    // int correctStackAnsCount = 0;
    // for(int i = csr->startNodeID ; i <= csr->endNodeID ; i ++){
    //     offset = i;//7
    //     printf("nodeID = %d\n", offset);

    //     computeBC_shareBased(csr, BCs);
    //     nextBFS = offset;
    //     printf("node = %d, Next_BFS = %d\n", offset, nextBFS);
    //     computeBC(csr, BCs2);
        
    //     check_SPandDist_Ans(csr);
    //     correctStackAnsCount += checkStackans(S_ori, S_method1, csr);
    //     for(int i = csr->startNodeID ; i <= csr->endNodeID ; i ++){
    //         free(S_ori->predecessors[i]->dataArr);
    //         free(S_method1->predecessors[i]->dataArr);
    //     }
    //     free(S_ori->predecessors);
    //     free(S_method1->predecessors);
    //     free(S_ori->nodeIDs);
    //     free(S_method1->nodeIDs);
    //     free(S_ori);
    //     free(S_method1);
    // }
    // printf("\nCorrect Stack Ans = %d\n", correctStackAnsCount);

    time2               = seconds();
    BrandesTime         = time2 - time1;
    printf("[Execution Time] %2f (s)\n", BrandesTime);
    //紀錄時間
    FILE *fptr = fopen("CostTime.txt", "a");
    if(fptr == NULL){
        printf("[Error] OpenFile : Output.txt\n");
        exit(1);
    }
    fprintf(fptr, "%s, %f\n", datasetPath, BrandesTime);
    fclose(fptr);
    //紀錄BC分數
    fptr = fopen("BC_Score.txt", "a");
    if(fptr == NULL){
        printf("[Error] OpenFile : Output.txt\n");
        exit(1);
    }
    fprintf(fptr, "%d %d\n", csr->startNodeID, csr->endNodeID);
    for(int node = csr->startNodeID ; node <= csr->endNodeID ; node ++){
        fprintf(fptr, "%f\n", BCs[node]);
    }
    fclose(fptr);
}