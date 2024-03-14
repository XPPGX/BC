/**
 * @todo 可以用普通的arr當queue，且size = csr->csrVSize
 * @todo d1Folding完成，接下來要寫個新function執行BC computation with d1Folding
 * @author XPPGX
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
// #define DEBUG_method2
#define Timing
// #define CheckAns
// #define Timing_Method1_And_Origin
// #define LiveJournal_Test
#pragma region globalVar
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

double method1_BC_time                          = 0;
double method1_forward_traverseTime             = 0;
double method1_additional_predecessor_handler   = 0;
double method1_backward_traverseTime            = 0;
double method1_memory_reset                     = 0;

double method2_BC_time                          = 0;
double ori_BC_time                              = 0;


int tempSourceID                                = -1;
#pragma endregion



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

inline void stackPushNode_shared(struct stack* _S, int _nodeID, int _prevTopDist, int _curNodeDist){
    // int prevTopNodeID = _S->nodeIDs[_S->top];

    _S->top ++;
    _S->nodeIDs[_S->top] = _nodeID;
    
    // if(_dist_arr_next[_nodeID] == _dist_arr_next[prevTopNodeID]){
    //     #ifdef DEBUG
    //     printf("S_next[%d] = %d, dist[%d] = %d\n", _S->top, _S->nodeIDs[_S->top], _S->nodeIDs[_S->top], _dist_arr_next[_S->nodeIDs[_S->top]]);
    //     #endif
    //     return;
    // } 
    
    // if(_dist_arr_next[_nodeID] < _dist_arr_next[prevTopNodeID]){
    //     #ifdef DEBUG
    //     printf("\tS_next[%d] = %d(%d), S_next[%d] = %d(%d), curDistIndex = %d, SWAP", _S->top, _S->nodeIDs[_S->top], _dist_arr_next[_S->nodeIDs[_S->top]], _S->top - 1, _S->nodeIDs[_S->top - 1], _dist_arr_next[_S->nodeIDs[_S->top - 1]], _S->S_curDistStartIndex);
    //     printf("(%d)<->(%d)\n", _S->nodeIDs[_S->top], _S->nodeIDs[_S->S_curDistStartIndex]);
    //     #endif
    //     swap(&(_S->nodeIDs[_S->top]), &(_S->nodeIDs[_S->S_curDistStartIndex]));
    //     _S->S_curDistStartIndex ++;
    //     return;
    // }
    // else if(_dist_arr_next[_nodeID] > _dist_arr_next[prevTopNodeID]){
    //     // _S->S_prevDistStartIndex = _S->S_curDistStartIndex;
    //     _S->S_curDistStartIndex = _S->top;
    //     #ifdef DEBUG
    //     printf("\tS_next.curIndex = %d\n", _S->S_curDistStartIndex);
    //     #endif
    // }
    if(_curNodeDist == _prevTopDist){
        #ifdef DEBUG
        // printf("S_next[%d] = %d, dist[%d] = %d\n", _S->top, _S->nodeIDs[_S->top], _S->nodeIDs[_S->top], _dist_arr_next[_S->nodeIDs[_S->top]]);
        #endif
        return;
    } 
    
    if(_curNodeDist < _prevTopDist){
        #ifdef DEBUG
        // printf("\tS_next[%d] = %d(%d), S_next[%d] = %d(%d), curDistIndex = %d, SWAP", _S->top, _S->nodeIDs[_S->top], _dist_arr_next[_S->nodeIDs[_S->top]], _S->top - 1, _S->nodeIDs[_S->top - 1], _dist_arr_next[_S->nodeIDs[_S->top - 1]], _S->S_curDistStartIndex);
        // printf("(%d)<->(%d)\n", _S->nodeIDs[_S->top], _S->nodeIDs[_S->S_curDistStartIndex]);
        #endif
        swap(&(_S->nodeIDs[_S->top]), &(_S->nodeIDs[_S->S_curDistStartIndex]));
        _S->S_curDistStartIndex ++;
        return;
    }
    else if(_curNodeDist > _prevTopDist){
        // _S->S_prevDistStartIndex = _S->S_curDistStartIndex;
        _S->S_curDistStartIndex = _S->top;
        #ifdef DEBUG
        // printf("\tS_next.curIndex = %d\n", _S->S_curDistStartIndex);
        #endif
    }
}

inline void stackPushNeighbor(struct stack* _S, int _nodeID, int _predecessor){
    vAppend(_S->predecessors[_nodeID], _predecessor);
}

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
    unsigned mask = 0x80000000U;
    unsigned value = *(unsigned*)v;
    while(mask){
        printf("%d", (mask&value)!=0U);
        mask >>= 1;
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
    
    
    #ifndef CheckAns
    // nextBFS = _csr->startNodeID;
    #endif

    #ifdef LiveJournal_Test
    double time1 = 0;
    double time2 = 0;
    #endif

    for(int sourceID = nextBFS; sourceID <= _csr->endNodeID ; sourceID ++){
        // if(_csr->csrNodesDegree[sourceID] == 1){
        //     break;
        // }
        #ifdef LiveJournal_Test
        time1 = seconds();
        #endif

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
        // for(int nodeIDIter = _csr->startNodeID ; nodeIDIter <= _csr->endNodeID ; nodeIDIter ++){
        //     printf("BC[%2d] = %2f\n", nodeIDIter, _BCs[nodeIDIter]);
        // }
        #ifdef LiveJournal_Test
        break;
        #endif
        
        #ifdef CheckAns
        break;
        #endif
    }

    #ifdef LiveJournal_Test
    time2 = seconds();
    printf("[Execution Time] Ori_BC(%d) = %f(s)\n", nextBFS, time2 - time1);
    ori_BC_time += time2 - time1;
    #endif

    #ifdef DEBUG
    for(int nodeIDIter = _csr->startNodeID ; nodeIDIter <= _csr->endNodeID ; nodeIDIter ++){
        printf("BC[%2d] = %2f\n", nodeIDIter, _BCs[nodeIDIter]);
    }
    #endif


    free(dependencies_arr);
    free(Q->dataArr);
    free(Q);

    #ifndef CheckAns
    free(numberOfSP_arr);
    free(dist_arr);
    for(int i = 0 ; i < _csr->csrVSize ; i ++){
        free(S->predecessors[i]->dataArr);
        free(S->predecessors[i]);
    }
    free(S->nodeIDs);
    free(S->predecessors);
    free(S);
    #else
    numberOfSP_arr_next_ori = numberOfSP_arr;
    dist_arr_next_ori       = dist_arr;
    S_ori                   = S;
    #endif
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
    double time1;
    double time2;

    register int testEndNodeID;
    #ifdef LiveJournal_Test
    testEndNodeID = _csr->startNodeID + 10000;
    #else
    testEndNodeID = _csr->endNodeID;
    #endif

    for(int sourceID = _csr->startNodeID ; sourceID <= testEndNodeID ; sourceID ++){
        if(nodesDone[sourceID] == 1){
            // printf("sourceID = %d done.\n", sourceID);
            continue;
        }
        #ifdef LiveJournal_Test
        time1 = seconds();
        #endif
        // printf("sourceID = %d start\t", sourceID);
        nodesDone[sourceID] = 1;
        
        //找自己的鄰居中，nodesDone != 1且degree最少的。
        minDegreeNeighborID = -1;
        // minDegree           = __INT_MAX__;
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
            if(_csr->csrNodesDegree[neighborID] != 1){
                // minDegree           = _csr->csrNodesDegree[neighborID];
                minDegreeNeighborID = neighborID;
                break;
            }
        }

        memset(numberOfSP_arr, 0, sizeof(float) * _csr->csrVSize);
        memset(dependencies_arr, 0, sizeof(float) * _csr->csrVSize);
        memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);
        
        resetQueue(Q);
        resetStack(S);

        if(minDegreeNeighborID != -1){
            //For test
            // nextBFS = minDegreeNeighborID;
            // printf("nextBFS = %d\n", nextBFS);
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
            // printf("\n");
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

                stackPushNode_shared(S_next, currentNodeID, dist_arr_next[S_next->nodeIDs[S_next->top]], dist_arr_next[currentNodeID]);

                //耗時最多的部分，在memcpy
                if(S_next->predecessors[currentNodeID]->size < tempVecs[0]->tail + 1){
                    resizeVec(S_next->predecessors[currentNodeID], (tempVecs[0]->tail + 1));
                }
                // resizeVec(S_next->predecessors[currentNodeID], (tempVecs[0]->tail + 1));
                memcpy(S_next->predecessors[currentNodeID]->dataArr, tempVecs[0]->dataArr, sizeof(int) * (tempVecs[0]->tail + 1));
                S_next->predecessors[currentNodeID]->tail = tempVecs[0]->tail;
                #ifdef DEBUG
                printf("\ttail = %d\n", S_next->predecessors[currentNodeID]->tail);
                #endif

                for(int succIndex = 0 ; succIndex <= tempVecs[2]->tail ; succIndex ++){
                    // successorID = tempVecs[2]->dataArr[succIndex];
                    #ifdef DEBUG
                    printf("\tnode %d add pred %d\n", successorID, currentNodeID);
                    #endif
                    stackPushNeighbor(S_next, tempVecs[2]->dataArr[succIndex], currentNodeID);
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
        // for(int i = 1 ; i <= S->top ; i ++){
        //     if(S->nodeIDs[i - 1] <= _csr->startNodeID){
        //         continue;
        //     }
        //     if(dist_arr[S->nodeIDs[i]] < dist_arr[S->nodeIDs[i - 1]]){
        //         printf("Stack1 Sequence wrong => S[i = %d] = %d, dist[%d] = %d, ", i, S->nodeIDs[i], S->nodeIDs[i], dist_arr[S->nodeIDs[i]]);
        //         printf("S[i - 1 = %d] = %d, dist[%d] = %d\n", i - 1, S->nodeIDs[i - 1], S->nodeIDs[i - 1], dist_arr[S->nodeIDs[i - 1]]);
        //         exit(1);
        //     }
        // }
        // if(minDegreeNeighborID != -1){
        //     for(int i = S_next->noSharedStartIndex ; i <= S_next->top ; i ++){
        //         if(i - 1 < S_next->noSharedStartIndex){
        //             continue;
        //         }
        //         if(dist_arr_next[S_next->nodeIDs[i]] < dist_arr_next[S_next->nodeIDs[i - 1]]){
        //             printf("Stack2 Sequence wrong => S[i = %d] = %d, dist[%d] = %d, ", i, S_next->nodeIDs[i], S_next->nodeIDs[i], dist_arr_next[S_next->nodeIDs[i]]);
        //             printf("S[i - 1 = %d] = %d, dist[%d] = %d\n", i - 1, S_next->nodeIDs[i - 1], S_next->nodeIDs[i - 1], dist_arr_next[S_next->nodeIDs[i - 1]]);
        //             exit(1);
        //         }
        //     }
        // }
#pragma endregion
        
#pragma region backwardTraverse
        struct vVector* predecessors    = NULL;
        int stackTopNodeID              = -1;
        int predecessorIndex            = -1;
        int predecessorID               = -1;
        // printf("source %d :\n", sourceID);
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
            memset(relations_next, 0, sizeof(int) * _csr->csrVSize);
        }
        #ifdef LiveJournal_Test
        time2 = seconds();
        printf("[Execution Time]method1_BC(%d, %d) = %f\n", sourceID, minDegreeNeighborID, time2 - time1);
        method1_BC_time += time2 - time1;
        #endif
#pragma endregion


#pragma region check_Value_Correctness
        // nextBFS = sourceID;
        #ifdef LiveJournal_Test
        nextBFS = sourceID;
        computeBC(_csr, BCs_ori);
        // printf("sourceID = %d, checking Ans...\n", sourceID);
        // check_SPandDist_Ans2(_csr, numberOfSP_arr, dist_arr);
        // checkStackans2(S_ori, S, _csr);
        nextBFS = minDegreeNeighborID;
        if(minDegreeNeighborID != -1){
        //     nextBFS = minDegreeNeighborID;
            computeBC(_csr, BCs_ori);
        //     printf("nextBFS = %d, checking Ans...\n", minDegreeNeighborID);
        //     check_SPandDist_Ans2(_csr, numberOfSP_arr_next, dist_arr_next);
        //     checkStackans2(S_ori, S_next, _csr);
        }
        #endif
#pragma endregion
        // break;
    }

    #ifdef LiveJournal_Test
    printf("[Execution Time] method1 : %f\n", method1_BC_time);
    printf("[Execution Time] Ori     : %f\n", ori_BC_time);
    #endif
    
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

void computeBC_shareBased2(struct CSR* _csr, float* _BCs){
    showCSR(_csr);
    #ifdef DEBUG_method2
    showCSR(_csr);
    #endif
    //malloc arrays for sourceID
    float* numberOfSP_arr   = (float*)calloc(sizeof(float), _csr->csrVSize);
    float* dependencies_arr = (float*)calloc(sizeof(float), _csr->csrVSize);
    int* dist_arr           = (int*)calloc(sizeof(int), _csr->csrVSize);
    struct stack* S         = initStack(_csr->csrVSize);
    //malloc arrays for nextBFS
    float* numberOfSP_arr_next      = (float*)calloc(sizeof(float), _csr->csrVSize);
    float* dependencies_arr_next    = (float*)calloc(sizeof(float), _csr->csrVSize);
    int* dist_arr_next              = (int*)calloc(sizeof(int), _csr->csrVSize);
    struct stack* S_next            = initStack(_csr->csrVSize);
    //for BFS traverse
    struct qQueue* Q        = InitqQueue();
    struct qQueue* Q_next   = InitqQueue();
    qInitResize(Q, _csr->csrVSize);
    qInitResize(Q_next, _csr->csrVSize);

    //When a nodes x forward and backward traverse are done, the nodesDone[x] = 1, else nodesDone[x] = 0
    int* nodesDone                  = (int*)calloc(sizeof(int), _csr->csrVSize);
    int* relations                  = (int*)calloc(sizeof(int), _csr->csrVSize);

    int minDegreeNeighborID = -1;
    int neighborID          = -1;
    
    #ifdef LiveJournal_Test
    double time1 = 0;
    double time2 = 0;
    #endif
    
    for(int sourceID = _csr->startNodeID ; sourceID <= _csr->endNodeID ; sourceID ++){
        
        if(nodesDone[sourceID] == 1){continue;}
        
        #ifdef LiveJournal_Test
        time1 = seconds();
        #endif

        nodesDone[sourceID] = 1;

        //reset arrays for sourceID
        memset(numberOfSP_arr, 0, sizeof(float) * _csr->csrVSize);
        memset(dependencies_arr, 0, sizeof(float) * _csr->csrVSize);
        memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);

        resetStack(S);
        resetQueue(Q);

        numberOfSP_arr[sourceID]    = 1;
        dist_arr[sourceID]          = 0;

        minDegreeNeighborID = -1;

#pragma region InitData
        //Find nextBfsID
        for(int neighborIndex = _csr->csrV[sourceID] ; neighborIndex < _csr->csrV[sourceID + 1] ; neighborIndex ++){
            neighborID = _csr->csrE[neighborIndex];
        
            if(minDegreeNeighborID == -1 && _csr->csrNodesDegree[neighborID] != 1 && nodesDone[neighborID] == 0){
                minDegreeNeighborID = neighborID;
                //for test
                nextBFS = minDegreeNeighborID;

                #ifdef DEBUG_method2
                printf("nextBFS = %d\n", minDegreeNeighborID);
                #endif

                nodesDone[minDegreeNeighborID]              = 1;

                //reset arrays for nextBfsID
                memset(numberOfSP_arr_next, 0, sizeof(float) * _csr->csrVSize);
                memset(dependencies_arr_next, 0, sizeof(float) * _csr->csrVSize);
                memset(dist_arr_next, -1, sizeof(int) * _csr->csrVSize);
                resetStack(S_next);
                resetQueue(Q_next);
                
                //init shared node info in nextBFS
                dist_arr_next[minDegreeNeighborID]          = 0;
                numberOfSP_arr_next[minDegreeNeighborID]    = 1;
                relations[minDegreeNeighborID]              = 1;
                //init sourceID info in nextBFS
                dist_arr_next[sourceID]                     = 1;
                // numberOfSP_arr_next[sourceID]               = 1;
                relations[sourceID]                         = 0;
                stackPushNode(S_next, sourceID);
                // stackPushNeighbor(S_next, sourceID, minDegreeNeighborID);

                //push minDegreeNeighborID into Q_next
                // qPushBack(Q, minDegreeNeighborID);
                qPushBack(Q_next, minDegreeNeighborID);

                //update info of minDegreeNeighborID in originBFS
                dist_arr[minDegreeNeighborID]       = 1;
                numberOfSP_arr[minDegreeNeighborID] = 1;
                stackPushNode(S, minDegreeNeighborID);
                stackPushNeighbor(S, minDegreeNeighborID, sourceID);

            }
            else{
                //push other neighbors into Q too
                // qPushBack(Q, neighborID);
                //update other neighbors info in originBFS
                dist_arr[neighborID]        = 1;
                #ifdef DEBUG_method2
                printf("dist[%d] = %d\n", neighborID, dist_arr[neighborID]);
                #endif
                numberOfSP_arr[neighborID]  = 1;
                stackPushNode(S, neighborID);
                stackPushNeighbor(S, neighborID, sourceID);
            }
        }

        if(minDegreeNeighborID != -1){ //When nextBFS is not empty
            for(int neighborIndex = _csr->csrV[minDegreeNeighborID] ; neighborIndex < _csr->csrV[minDegreeNeighborID + 1] ; neighborIndex ++){
                neighborID = _csr->csrE[neighborIndex];
                if(neighborID == sourceID){continue;}
                if(dist_arr[neighborID] == 1){
                    qPushBack(Q_next, neighborID);
                    stackPushNode(S_next, neighborID);
                    relations[neighborID]       = 1;
                    dist_arr_next[neighborID]   = 1;
                }
            }
            for(int neighborIndex = _csr->csrV[sourceID] ; neighborIndex < _csr->csrV[sourceID + 1] ; neighborIndex ++){
                neighborID = _csr->csrE[neighborIndex];
                if(dist_arr_next[neighborID] == -1){
                    qPushBack(Q, neighborID);
                }
            }
        }
        else{ //when nextBFS is empty
            for(int neighborIndex = _csr->csrV[sourceID] ; neighborIndex < _csr->csrV[sourceID + 1] ; neighborIndex ++){
                neighborID = _csr->csrE[neighborIndex];
                qPushBack(Q, neighborID);
            }
        }
#pragma endregion //InitData
        
        #ifdef DEBUG_method2
        printf("Q :\n");
        for(int i = 0 ; i <= Q->rear ; i ++){
            printf("Q[%d] = %d\n", i, Q->dataArr[i]);
        }
        printf("\n");
        printf("Q_next :\n");
        for(int i = 0 ; i <= Q_next->rear ; i ++){
            printf("Q_next[%d] = %d\n", i, Q_next->dataArr[i]);
        }
        printf("\n");

        printf("S : \n");
        for(int i = S->top ; i > -1 ; i --){
            printf("S[%d] = %2d\n", i, S->nodeIDs[i]);
        }
        printf("\n");
        
        printf("S_next : \n");
        for(int i = S_next->top ; i > -1 ; i --){
            printf("S_next[%d] = %2d\n", i, S_next->nodeIDs[i]);
        }
        printf("\n");
        
        #endif
        
        
        //If Q is empty, it means the nextBFS(source) is not found.
        //Thus, the sourceID should be push into Q to perform ordinary BFS.
        // if(qIsEmpty(Q)){
        //     printf("Q is empty at beginning!!\n");
        //     qPushBack(Q, sourceID);
        // }
        
        //Forward Traverse
        register int currentNodeID  = -1;
        register int neighborNodeID = -1;

        // int checkSource = -1;
        // int test1       = -1;
        // int test2       = -1;
        // int test3       = -1;
        #ifdef CheckAns
        printf("source = %2d, nextBFS = %2d\n", sourceID, minDegreeNeighborID);
        #endif

#pragma region forwardTraverse
        if(minDegreeNeighborID != -1){ //When nextBFS is not empty
            register int tempRear       = -1;
            register int tempQ_RearID   = -1;

            while(!qIsEmpty(Q_next) || !qIsEmpty(Q)){

                tempQ_RearID = Q->dataArr[Q->rear];
                while(!qIsEmpty(Q_next)){
                    currentNodeID = qPopFront(Q_next);

                    if(dist_arr[currentNodeID] > dist_arr[tempQ_RearID] && !qIsEmpty(Q)){
                        //recovery Q_next->front
                        Q_next->front --;

                        #ifdef DEBUG_method2
                        printf("dist[%2d] = %2d > dist[%2d] = %2d\n", currentNodeID, dist_arr[currentNodeID], tempQ_RearID, dist_arr[tempQ_RearID]);
                        #endif

                        break;
                    }

                    #ifdef DEBUG_method2
                    printf("currentID = %2d, ", currentNodeID);
                    printf("relation = %2d, dist = %2d, dist_next = %2d, SP = %2f, SP_next = %2f\n", relations[currentNodeID], dist_arr[currentNodeID], dist_arr_next[currentNodeID], numberOfSP_arr[currentNodeID], numberOfSP_arr_next[currentNodeID]);
                    #endif

                    for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
                        neighborNodeID = _csr->csrE[neighborIndex];

                        #ifdef DEBUG_method2
                        printf("\tneighbor %2d, relation = %2d, dist = %2d, dist_next = %2d, SP = %2f, SP_next = %2f\n", neighborNodeID, relations[neighborNodeID], dist_arr[neighborNodeID], dist_arr_next[neighborNodeID], numberOfSP_arr[neighborNodeID], numberOfSP_arr_next[neighborNodeID]);
                        #endif
                        
                        // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                        //     printf("=======\n");
                        //     printf("[Q_N]currentID  = %d, relation = %d, dist = %d, dist_next = %d, SP = %f, SP_next %f\n", currentNodeID, relations[currentNodeID], dist_arr[currentNodeID], dist_arr_next[currentNodeID], numberOfSP_arr[currentNodeID], numberOfSP_arr_next[currentNodeID]);
                        //     printf("neighborID = %d, relation = %d, dist = %d, dist_next = %d, SP = %f, SP_next %f\n", neighborNodeID, relations[neighborNodeID], dist_arr[neighborNodeID], dist_arr_next[neighborNodeID], numberOfSP_arr[neighborNodeID], numberOfSP_arr_next[neighborNodeID]);
                        // }

                        if(dist_arr[neighborNodeID] == -1){
                            qPushBack(Q_next, neighborNodeID);
                            relations[neighborNodeID]       = 1;
                            dist_arr[neighborNodeID]        = dist_arr[currentNodeID] + 1;
                            dist_arr_next[neighborNodeID]   = dist_arr_next[currentNodeID] + 1;
                            
                            stackPushNode(S, neighborNodeID);
                            stackPushNode(S_next, neighborNodeID);

                            #ifdef DEBUG_method2
                            printf("\t\t[Push(Qn)] %2d", neighborNodeID);
                            printf("relation = %2d, dist = %2d, dist_next = %2d\n", relations[neighborNodeID], dist_arr[neighborNodeID], dist_arr_next[neighborNodeID]);
                            printf("\t\t[Push(S) ] neighbor %2d dist = %2d\n", neighborNodeID, dist_arr[neighborNodeID]);
                            printf("\t\t[Push(Sn)] neighbor %2d dist_next = %2d\n", neighborNodeID, dist_arr_next[neighborNodeID]);

                            #endif

                            numberOfSP_arr[neighborNodeID] += numberOfSP_arr[currentNodeID];
                            stackPushNeighbor(S, neighborNodeID, currentNodeID);

                            #ifdef DEBUG_method2
                            printf("\t\t[Update 0] node %2d, relation = %2d, ", neighborNodeID, relations[neighborNodeID]);
                            printf("SP = %2f, add Pred %2d\n", numberOfSP_arr[neighborNodeID], currentNodeID);
                            #endif

                            numberOfSP_arr_next[neighborNodeID] += numberOfSP_arr_next[currentNodeID];
                            stackPushNeighbor(S_next, neighborNodeID, currentNodeID);

                            #ifdef DEBUG_method2
                            printf("\t\t[Update 1] node %2d, relation = %2d, ", neighborNodeID, relations[neighborNodeID]);
                            printf("SP_next = %2f, add Pred_next %2d\n", numberOfSP_arr_next[neighborNodeID], currentNodeID);
                            #endif
                            // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                            //     printf("\t[Push to Q_next]\n");
                            // }
                        }
                        else{
                            if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] && (dist_arr_next[neighborNodeID] == -1 || dist_arr_next[currentNodeID] + 1 < dist_arr_next[neighborNodeID])){
                                
                                dist_arr_next[neighborNodeID]               = dist_arr_next[currentNodeID] + 1;

                                numberOfSP_arr_next[neighborNodeID]         = 0;

                                S_next->predecessors[neighborNodeID]->tail  = -1;
                                
                                if(dist_arr_next[neighborNodeID] == dist_arr[neighborNodeID]){
                                    stackPushNode(S_next, neighborNodeID);
                                }

                                #ifdef DEBUG_method2
                                printf("\t\tFind same level black node %2d\n", neighborNodeID);
                                printf("\t\t[Reset_next] %2d reset next info, and dist_next = %2d\n", neighborNodeID, dist_arr_next[neighborNodeID]);
                                printf("\t\t[Push(Sn)] neighbor %2d dist_next = %2d\n", neighborNodeID,  dist_arr_next[neighborNodeID]);
                                #endif

                                numberOfSP_arr_next[neighborNodeID] += numberOfSP_arr_next[currentNodeID];
                                stackPushNeighbor(S_next, neighborNodeID, currentNodeID);

                                #ifdef DEBUG_method2
                                printf("\t\t[Update 1] node %2d, relation = %2d, ", neighborNodeID, relations[neighborNodeID]);
                                printf("SP_next = %2f, add Pred_next %2d\n", numberOfSP_arr_next[neighborNodeID], currentNodeID);
                                #endif
                                // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                                //     printf("\t[Find same level black node] Reset %2d next info\n", neighborNodeID);
                                // }
                            }
                            else{
                                if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){
                                    numberOfSP_arr[neighborNodeID] += numberOfSP_arr[currentNodeID];
                                    stackPushNeighbor(S, neighborNodeID, currentNodeID);

                                    #ifdef DEBUG_method2
                                    printf("\t\t[Update 0] node %2d, relation = %2d, ", neighborNodeID, relations[neighborNodeID]);
                                    printf("SP = %2f, add Pred %2d\n", numberOfSP_arr[neighborNodeID], currentNodeID);
                                    #endif

                                    // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                                    //     printf("\t[Update 0]\n");
                                    // }
                                }

                                if(dist_arr_next[neighborNodeID] == dist_arr_next[currentNodeID] + 1){
                                    numberOfSP_arr_next[neighborNodeID] += numberOfSP_arr_next[currentNodeID];
                                    stackPushNeighbor(S_next, neighborNodeID, currentNodeID);

                                    #ifdef DEBUG_method2
                                    printf("\t\t[Update 1] node %2d, relation = %2d, ", neighborNodeID, relations[neighborNodeID]);
                                    printf("SP_next = %2f, add Pred_next %2d\n", numberOfSP_arr_next[neighborNodeID], currentNodeID);
                                    #endif

                                    // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                                    //     printf("\t[Update 1]\n");
                                    // }
                                }
                            }
                        } 
                    }
                }
                
                #ifdef DEBUG_method2
                printf("\nQ_next break!!!!!!!!!!!!!!!!!!\n");
                #endif

                tempRear = Q->rear + 1;
                while(Q->front < tempRear){
                    currentNodeID = qPopFront(Q);

                    #ifdef DEBUG_method2
                    printf("currentID = %2d, ", currentNodeID);
                    printf("relation = %2d, dist = %2d, dist_next = %2d, SP = %2f, SP_next = %2f\n", relations[currentNodeID], dist_arr[currentNodeID], dist_arr_next[currentNodeID], numberOfSP_arr[currentNodeID], numberOfSP_arr_next[currentNodeID]);
                    #endif

                    if(dist_arr[currentNodeID] == dist_arr_next[currentNodeID]){
                        relations[currentNodeID] = 2;
                        
                        #pragma region rearrangeIf_1
                        for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
                            neighborNodeID = _csr->csrE[neighborIndex];

                            #ifdef DEBUG_method2
                            printf("\tneighbor %2d, relation = %2d, dist = %2d, dist_next = %2d, SP = %2f, SP_next = %2f\n", neighborNodeID, relations[neighborNodeID], dist_arr[neighborNodeID], dist_arr_next[neighborNodeID], numberOfSP_arr[neighborNodeID], numberOfSP_arr_next[neighborNodeID]);
                            #endif

                            if(relations[neighborNodeID] == 0 && (dist_arr[neighborNodeID] != dist_arr_next[neighborNodeID]) && (dist_arr_next[currentNodeID] == dist_arr_next[neighborNodeID] + 1)){ //pull_update black : when dist_next[current] == dist_next[neighbor]
                                numberOfSP_arr_next[currentNodeID] += numberOfSP_arr_next[neighborNodeID];
                                stackPushNeighbor(S_next, currentNodeID, neighborNodeID);
                                
                                #ifdef DEBUG_method2
                                printf("\t\t[Update 2] node %2d, SP_next = %2f, add Pred_next %2d\n", currentNodeID, numberOfSP_arr_next[currentNodeID], neighborNodeID);
                                #endif
                                // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                                //     printf("\t[Update 2]\n");
                                // }
                            }
                            else if(relations[neighborNodeID] == 1 && (dist_arr_next[currentNodeID] + 1 == dist_arr_next[neighborNodeID])){ //push_update red
                                numberOfSP_arr_next[neighborNodeID] += numberOfSP_arr_next[currentNodeID];
                                stackPushNeighbor(S_next, neighborNodeID, currentNodeID);
                                
                                #ifdef DEBUG_method2
                                printf("\t\t[Update 4] node %2d, SP_next = %2f, add Pred_next %2d\n", neighborNodeID, numberOfSP_arr_next[neighborNodeID], currentNodeID);
                                #endif
                                // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                                //     printf("\t[Update 4]\n");
                                // }
                            }
                            else if(dist_arr[neighborNodeID] == -1){
                                qPushBack(Q, neighborNodeID);
                                dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;
                                stackPushNode(S, neighborNodeID);

                                #ifdef DEBUG_method2
                                printf("\t\t[Push] %2d to Q, ", neighborNodeID);
                                printf("relation = %2d, dist = %2d, dist_next = %2d\n", relations[neighborNodeID], dist_arr[neighborNodeID], dist_arr_next[neighborNodeID]);
                                printf("\t\t[Push(S) ] neighbor %2d, dist = %2d\n", neighborNodeID, dist_arr[neighborNodeID]);
                                #endif
                                // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                                //     printf("\t[Push to Q]\n");
                                // }
                            }

                            if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){
                                numberOfSP_arr[neighborNodeID] += numberOfSP_arr[currentNodeID];
                                stackPushNeighbor(S, neighborNodeID, currentNodeID);
                                
                                #ifdef DEBUG_method2
                                printf("\t\t[Update 3] node %2d, SP = %2f, add Pred %2d\n", neighborNodeID, numberOfSP_arr[neighborNodeID], currentNodeID);
                                #endif

                                // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                                //     printf("\t[Update 3]\n");
                                // }
                            }

                            if(dist_arr_next[neighborNodeID] == -1){
                                dist_arr_next[neighborNodeID] = dist_arr_next[currentNodeID] + 1;

                                #ifdef DEBUG_method2
                                printf("\t\t[Update 5] node %2d, relation = %2d, dist_next = %2d\n", neighborNodeID, relations[neighborNodeID], dist_arr_next[neighborNodeID]);
                                #endif

                                if(dist_arr_next[neighborNodeID] == dist_arr[neighborNodeID]){
                                    stackPushNode(S_next, neighborNodeID);
                                    
                                    #ifdef DEBUG_method2
                                    printf("\t\t[Push(Sn)] neighbor %2d, dist_next = %2d\n", neighborNodeID, dist_arr_next[neighborNodeID]);
                                    #endif
                                }

                                numberOfSP_arr_next[neighborNodeID] += numberOfSP_arr_next[currentNodeID];
                                stackPushNeighbor(S_next, neighborNodeID, currentNodeID);

                                #ifdef DEBUG_method2
                                printf("\t\t[Update 6] node %2d, SP_next = %2f, add Pred %2d\n", neighborNodeID, numberOfSP_arr_next[neighborNodeID], currentNodeID);
                                #endif
                                // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                                //     printf("\t[Update 5]\n");
                                // }
                            }

                            else if(relations[neighborNodeID] == 0 && (dist_arr_next[neighborNodeID] == dist_arr_next[currentNodeID] + 1)){
                                numberOfSP_arr_next[neighborNodeID] += numberOfSP_arr_next[currentNodeID];
                                stackPushNeighbor(S_next, neighborNodeID, currentNodeID);

                                #ifdef DEBUG_method2
                                printf("\t\t[Update 6] node %2d, SP_next = %2f, add Pred %2d\n", neighborNodeID, numberOfSP_arr_next[neighborNodeID], currentNodeID);
                                #endif
                                // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                                //     printf("\t[Update 6]\n");
                                // }
                            }

                        }
                        #pragma endregion
                        
                        
                    }
                    else{
                        stackPushNode(S_next, currentNodeID);

                        #ifdef DEBUG_method2
                        printf("\t\t[Push(Sn)] current %2d dist_next = %2d\n", currentNodeID, dist_arr_next[currentNodeID]);
                        #endif

                        #pragma region rearrangeIf_2
                        for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
                            neighborNodeID = _csr->csrE[neighborIndex];

                            #ifdef DEBUG_method2
                            printf("\tneighbor %2d, relation = %2d, dist = %2d, dist_next = %2d, SP = %2f, SP_next = %2f\n", neighborNodeID, relations[neighborNodeID], dist_arr[neighborNodeID], dist_arr_next[neighborNodeID], numberOfSP_arr[neighborNodeID], numberOfSP_arr_next[neighborNodeID]);
                            #endif
                            
                            // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3) ){
                            //     printf("======\n");
                            //     printf("[Q_S]currentID  = %d, relation = %d, dist = %d, dist_next = %d, SP = %f, SP_next %f\n", currentNodeID, relations[currentNodeID], dist_arr[currentNodeID], dist_arr_next[currentNodeID], numberOfSP_arr[currentNodeID], numberOfSP_arr_next[currentNodeID]);
                            //     printf("neighborID = %d, relation = %d, dist = %d, dist_next = %d, SP = %f, SP_next %f\n", neighborNodeID, relations[neighborNodeID], dist_arr[neighborNodeID], dist_arr_next[neighborNodeID], numberOfSP_arr[neighborNodeID], numberOfSP_arr_next[neighborNodeID]);
                            // }

                            if(dist_arr_next[neighborNodeID] != -1 && (dist_arr_next[currentNodeID] == -1 || dist_arr_next[currentNodeID] > dist_arr_next[neighborNodeID] + 1)){
                                //set relations
                                // relations[currentNodeID]                   = 2;
                                //set dist_next[currentNodeID]
                                dist_arr_next[currentNodeID]                = dist_arr_next[neighborNodeID] + 1;
                                //reset SP_next[currentNodeID]
                                numberOfSP_arr_next[currentNodeID]          = 0;
                                //reset predecessors vector of currentNodeID
                                S_next->predecessors[currentNodeID]->tail   = -1;

                                #ifdef DEBUG_method2
                                printf("\t\t[Reset_next] %2d reset next info, and dist_next = %2d\n", currentNodeID, dist_arr_next[currentNodeID]);
                                #endif

                                // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                                //     printf("\t[Reset_next]\n");
                                // }
                            }
                            
                            if(relations[neighborNodeID] == 0 && (dist_arr[neighborNodeID] != dist_arr_next[neighborNodeID]) && (dist_arr_next[currentNodeID] == dist_arr_next[neighborNodeID] + 1)){ //pull_update black : when dist_next[current] == dist_next[neighbor]
                                numberOfSP_arr_next[currentNodeID] += numberOfSP_arr_next[neighborNodeID];
                                stackPushNeighbor(S_next, currentNodeID, neighborNodeID);
                                
                                #ifdef DEBUG_method2
                                printf("\t\t[Update 2] node %2d, SP_next = %2f, add Pred_next %2d\n", currentNodeID, numberOfSP_arr_next[currentNodeID], neighborNodeID);
                                #endif
                                // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                                //     printf("\t[Update 2]\n");
                                // }
                            }
                            else if(relations[neighborNodeID] == 1 && (dist_arr_next[currentNodeID] + 1 == dist_arr_next[neighborNodeID])){ //push_update red
                                numberOfSP_arr_next[neighborNodeID] += numberOfSP_arr_next[currentNodeID];
                                stackPushNeighbor(S_next, neighborNodeID, currentNodeID);
                                
                                #ifdef DEBUG_method2
                                printf("\t\t[Update 4] node %2d, SP_next = %2f, add Pred_next %2d\n", neighborNodeID, numberOfSP_arr_next[neighborNodeID], currentNodeID);
                                #endif
                                // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                                //     printf("\t[Update 4]\n");
                                // }
                            }

                            if(dist_arr[neighborNodeID] == -1){
                                qPushBack(Q, neighborNodeID);
                                dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;
                                stackPushNode(S, neighborNodeID);

                                #ifdef DEBUG_method2
                                printf("\t\t[Push] %2d to Q, ", neighborNodeID);
                                printf("relation = %2d, dist = %2d, dist_next = %2d\n", relations[neighborNodeID], dist_arr[neighborNodeID], dist_arr_next[neighborNodeID]);
                                printf("\t\t[Push(S) ] neighbor %2d, dist = %2d\n", neighborNodeID, dist_arr[neighborNodeID]);
                                #endif

                                numberOfSP_arr[neighborNodeID] += numberOfSP_arr[currentNodeID];
                                stackPushNeighbor(S, neighborNodeID, currentNodeID);
                                
                                #ifdef DEBUG_method2
                                printf("\t\t[Update 3] node %2d, SP = %2f, add Pred %2d\n", neighborNodeID, numberOfSP_arr[neighborNodeID], currentNodeID);
                                #endif
                                // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                                //     printf("\t[Push to Q]\n");
                                // }
                            }
                            else if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){
                                numberOfSP_arr[neighborNodeID] += numberOfSP_arr[currentNodeID];
                                stackPushNeighbor(S, neighborNodeID, currentNodeID);
                                
                                #ifdef DEBUG_method2
                                printf("\t\t[Update 3] node %2d, SP = %2f, add Pred %2d\n", neighborNodeID, numberOfSP_arr[neighborNodeID], currentNodeID);
                                #endif

                                // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                                //     printf("\t[Update 3]\n");
                                // }
                            }
                            #pragma endregion
                        }
                        #pragma endregion //rearrangeIf_2
                    }

                    /*
                    #pragma region correctIfLogic
                    for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
                        neighborNodeID = _csr->csrE[neighborIndex];

                        

                        #ifdef DEBUG_method2
                        printf("\tneighbor %2d, relation = %2d, dist = %2d, dist_next = %2d, SP = %2f, SP_next = %2f\n", neighborNodeID, relations[neighborNodeID], dist_arr[neighborNodeID], dist_arr_next[neighborNodeID], numberOfSP_arr[neighborNodeID], numberOfSP_arr_next[neighborNodeID]);
                        #endif
                        
                        // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3) ){
                        //     printf("======\n");
                        //     printf("[Q_S]currentID  = %d, relation = %d, dist = %d, dist_next = %d, SP = %f, SP_next %f\n", currentNodeID, relations[currentNodeID], dist_arr[currentNodeID], dist_arr_next[currentNodeID], numberOfSP_arr[currentNodeID], numberOfSP_arr_next[currentNodeID]);
                        //     printf("neighborID = %d, relation = %d, dist = %d, dist_next = %d, SP = %f, SP_next %f\n", neighborNodeID, relations[neighborNodeID], dist_arr[neighborNodeID], dist_arr_next[neighborNodeID], numberOfSP_arr[neighborNodeID], numberOfSP_arr_next[neighborNodeID]);
                        // }

                        if(dist_arr_next[neighborNodeID] != -1 && (dist_arr_next[currentNodeID] == -1 || dist_arr_next[currentNodeID] > dist_arr_next[neighborNodeID] + 1)){
                            //set relations
                            // relations[currentNodeID]                   = 2;
                            //set dist_next[currentNodeID]
                            dist_arr_next[currentNodeID]                = dist_arr_next[neighborNodeID] + 1;
                            //reset SP_next[currentNodeID]
                            numberOfSP_arr_next[currentNodeID]          = 0;
                            //reset predecessors vector of currentNodeID
                            S_next->predecessors[currentNodeID]->tail   = -1;

                            #ifdef DEBUG_method2
                            printf("\t\t[Reset_next] %2d reset next info, and dist_next = %2d\n", currentNodeID, dist_arr_next[currentNodeID]);
                            #endif

                            // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                            //     printf("\t[Reset_next]\n");
                            // }
                        }
                        
                        if(relations[neighborNodeID] == 0 && (dist_arr[neighborNodeID] != dist_arr_next[neighborNodeID]) && (dist_arr_next[currentNodeID] == dist_arr_next[neighborNodeID] + 1)){ //pull_update black : when dist_next[current] == dist_next[neighbor]
                            numberOfSP_arr_next[currentNodeID] += numberOfSP_arr_next[neighborNodeID];
                            stackPushNeighbor(S_next, currentNodeID, neighborNodeID);
                            
                            #ifdef DEBUG_method2
                            printf("\t\t[Update 2] node %2d, SP_next = %2f, add Pred_next %2d\n", currentNodeID, numberOfSP_arr_next[currentNodeID], neighborNodeID);
                            #endif
                            // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                            //     printf("\t[Update 2]\n");
                            // }
                        }
                        else if(relations[neighborNodeID] == 1 && (dist_arr_next[currentNodeID] + 1 == dist_arr_next[neighborNodeID])){ //push_update red
                            numberOfSP_arr_next[neighborNodeID] += numberOfSP_arr_next[currentNodeID];
                            stackPushNeighbor(S_next, neighborNodeID, currentNodeID);
                            
                            #ifdef DEBUG_method2
                            printf("\t\t[Update 4] node %2d, SP_next = %2f, add Pred_next %2d\n", neighborNodeID, numberOfSP_arr_next[neighborNodeID], currentNodeID);
                            #endif
                            // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                            //     printf("\t[Update 4]\n");
                            // }
                        }

                        if(dist_arr[neighborNodeID] == -1){
                            qPushBack(Q, neighborNodeID);
                            dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;
                            stackPushNode(S, neighborNodeID);

                            #ifdef DEBUG_method2
                            printf("\t\t[Push] %2d to Q, ", neighborNodeID);
                            printf("relation = %2d, dist = %2d, dist_next = %2d\n", relations[neighborNodeID], dist_arr[neighborNodeID], dist_arr_next[neighborNodeID]);
                            printf("\t\t[Push(S) ] neighbor %2d, dist = %2d\n", neighborNodeID, dist_arr[neighborNodeID]);
                            #endif
                            // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                            //     printf("\t[Push to Q]\n");
                            // }
                        }

                        if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){
                            numberOfSP_arr[neighborNodeID] += numberOfSP_arr[currentNodeID];
                            stackPushNeighbor(S, neighborNodeID, currentNodeID);
                            
                            #ifdef DEBUG_method2
                            printf("\t\t[Update 3] node %2d, SP = %2f, add Pred %2d\n", neighborNodeID, numberOfSP_arr[neighborNodeID], currentNodeID);
                            #endif

                            // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                            //     printf("\t[Update 3]\n");
                            // }
                        }

                        #pragma region newCode
                        if(relations[currentNodeID] == 2){

                            if(dist_arr_next[neighborNodeID] == -1){
                                dist_arr_next[neighborNodeID] = dist_arr_next[currentNodeID] + 1;

                                #ifdef DEBUG_method2
                                printf("\t\t[Update 5] node %2d, relation = %2d, dist_next = %2d\n", neighborNodeID, relations[neighborNodeID], dist_arr_next[neighborNodeID]);
                                #endif

                                if(dist_arr_next[neighborNodeID] == dist_arr[neighborNodeID]){
                                    stackPushNode(S_next, neighborNodeID);
                                    
                                    #ifdef DEBUG_method2
                                    printf("\t\t[Push(Sn)] neighbor %2d, dist_next = %2d\n", neighborNodeID, dist_arr_next[neighborNodeID]);
                                    #endif
                                }

                                // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                                //     printf("\t[Update 5]\n");
                                // }
                            }

                            if(relations[neighborNodeID] == 0 && (dist_arr_next[neighborNodeID] == dist_arr_next[currentNodeID] + 1)){
                                numberOfSP_arr_next[neighborNodeID] += numberOfSP_arr_next[currentNodeID];
                                stackPushNeighbor(S_next, neighborNodeID, currentNodeID);

                                #ifdef DEBUG_method2
                                printf("\t\t[Update 6] node %2d, SP_next = %2f, add Pred %2d\n", neighborNodeID, numberOfSP_arr_next[neighborNodeID], currentNodeID);
                                #endif
                                // if(minDegreeNeighborID == checkSource && (neighborNodeID == test1 || neighborNodeID == test2 || neighborNodeID == test3 || currentNodeID == test1 || currentNodeID == test2 || currentNodeID == test3)){
                                //     printf("\t[Update 6]\n");
                                // }
                            }
                        }
                        #pragma endregion
                    }
                    #pragma endregion //correctIfLogic
                    */
                }
                
                #ifdef DEBUG_method2
                printf("\nQ break!!!!!!!!!!!!!!!!!!!!!!\n");
                #endif

            }//outer while loop
            
            memset(relations, 0, sizeof(int) * _csr->csrVSize);

        }
        else{ //When nextBFS is empty
            while(!qIsEmpty(Q)){
                currentNodeID = qPopFront(Q);
                
                // printf("currentNodeID %2d : S[%2d] = %2d, dist = %2d\n", currentNodeID, S->top, S->nodeIDs[S->top], dist_arr[S->nodeIDs[S->top]]);

                for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborNodeID = _csr->csrE[neighborIndex];
                    if(dist_arr[neighborNodeID] < 0){
                        qPushBack(Q, neighborNodeID);
                        // stackPushNode(S, neighborNodeID);
                        // printf("[Push(S)] %2d\n", neighborNodeID);
                        dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;
                    }

                    if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){
                        numberOfSP_arr[neighborNodeID] += numberOfSP_arr[currentNodeID];
                        stackPushNeighbor(S, neighborNodeID, currentNodeID);
                    }
                }
            }
        }
#pragma endregion


        #ifdef DEBUG_method2
        printf("S : node(dist)\n");
        for(int i = S->top ; i > -1 ; i --){
            printf("S[%2d] = %2d(%2d)\n", i, S->nodeIDs[i], dist_arr[S->nodeIDs[i]]);
        }
        printf("\n");
        printf("S_next : node(dist_next)\n");
        for(int i = S_next->top ; i > -1 ; i --){
            printf("S[%2d] = %2d(%2d)\n", i, S_next->nodeIDs[i], dist_arr_next[S_next->nodeIDs[i]]);
        }
        printf("\n");
        #endif
#pragma region checkAns
        #ifdef CheckAns

        if(minDegreeNeighborID != -1){
            printf("Check S sequence...");
            checkStackSequence(_csr, S, dist_arr, sourceID);
            nextBFS = sourceID;
            computeBC(_csr, BCs_ori);
            checkStackans2(S_ori, S, _csr);
            check_SPandDist_Ans2(_csr, numberOfSP_arr, dist_arr);

            printf("Check S_next sequence...");
            checkStackSequence(_csr, S_next, dist_arr_next, minDegreeNeighborID);
            nextBFS = minDegreeNeighborID;
            computeBC(_csr, BCs_ori);
            checkStackans2(S_ori, S_next, _csr);
            check_SPandDist_Ans2(_csr, numberOfSP_arr_next, dist_arr_next);
        }
        
        // break; 
        #endif
#pragma endregion //checkAns

#pragma region backwardTraverse

        struct vVector* predecessors    = NULL;
        int stackTopNodeID              = -1;
        int predecessorIndex            = -1;
        int predecessorID               = -1;
        if(minDegreeNeighborID != -1){
            while(!stackIsEmpty(S)){
                stackPop(S, &predecessors, &stackTopNodeID);
                for(predecessorIndex = 0 ; predecessorIndex <= predecessors->tail ; predecessorIndex ++){
                    predecessorID = predecessors->dataArr[predecessorIndex];

                    dependencies_arr[predecessorID] += ((numberOfSP_arr[predecessorID] / numberOfSP_arr[stackTopNodeID]) * (1 + dependencies_arr[stackTopNodeID]));
                    
                    _BCs[stackTopNodeID]            += dependencies_arr[stackTopNodeID];
                }
            }
            while(!stackIsEmpty(S_next)){
                stackPop(S_next, &predecessors, &stackTopNodeID);
                for(predecessorIndex = 0 ; predecessorIndex <= predecessors->tail ; predecessorIndex ++){
                    predecessorID = predecessors->dataArr[predecessorIndex];
                    
                    dependencies_arr_next[predecessorID] += ((numberOfSP_arr_next[predecessorID] / numberOfSP_arr_next[stackTopNodeID]) * (1 + dependencies_arr_next[stackTopNodeID]));
                    
                    _BCs[stackTopNodeID] += dependencies_arr_next[stackTopNodeID];
                }
            }
        }
        else{
            int nodeID = -1;
            for(int Q_backIter = Q->rear ; Q_backIter > -1 ; Q_backIter --){
                nodeID = Q->dataArr[Q_backIter];
                predecessors = S->predecessors[nodeID];
                for(predecessorIndex = 0 ; predecessorIndex <= predecessors->tail ; predecessorIndex ++){
                    predecessorID = predecessors->dataArr[predecessorIndex];

                    dependencies_arr[predecessorID] += ((numberOfSP_arr[predecessorID] / numberOfSP_arr[nodeID]) * (1 + dependencies_arr[nodeID]));

                    _BCs[nodeID] += dependencies_arr[nodeID];
                }
            }
        }
        
#pragma endregion //backwardTraverse

        #ifdef LiveJournal_Test
        time2 = seconds();
        printf("[Execution Time] method2_BC(%d, %d) = %f(s)\n", sourceID, minDegreeNeighborID, time2 - time1);
        method2_BC_time += time2 - time1;

        //origin computeBC
        nextBFS = sourceID;
        computeBC(_csr, BCs_ori);
        if(minDegreeNeighborID != -1){
            nextBFS = minDegreeNeighborID;
            computeBC(_csr, BCs_ori);
        }
        #endif
        // break;

    }//The most outer for loop

    
    free(numberOfSP_arr);
    free(dependencies_arr);
    free(dist_arr);
    free(numberOfSP_arr_next);
    free(dependencies_arr_next);
    free(dist_arr_next);
    free(Q->dataArr);
    free(Q);
    free(Q_next->dataArr);
    free(Q_next);
    free(relations);
    free(nodesDone);
    for(int i = 0 ; i < _csr->csrVSize ; i ++){
        free(S->predecessors[i]->dataArr);
        free(S->predecessors[i]);
        free(S_next->predecessors[i]->dataArr);
        free(S_next->predecessors[i]);
    }
    free(S->predecessors);
    free(S->nodeIDs);
    free(S_next->nodeIDs);
    free(S_next->predecessors);
    free(S);
    free(S_next);
}

int* computeCC(struct CSR* _csr, float* _CCs){
    // showCSR(_csr);
    int* dist_arr = (int*)calloc(sizeof(int), _csr->csrVSize);

    struct qQueue* Q = InitqQueue();
    qInitResize(Q, _csr->csrVSize);

    double time1;
    double time2;

    int sourceID;
    time1 = seconds();

    #ifdef CheckAns
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
            _CCs[nodeID] = _CCs[nodeID] + (float)dist_arr[nodeID];
            // printf("CC[%2d] = %2f\n", nodeID, _CCs[nodeID]);
        }
        #ifdef CheckAns
        break;
        #endif
    }
    time2 = seconds();
    printf("[ExecutionTime] %2f\n", time2 - time1);
    
    
    free(Q->dataArr);
    free(Q);

    // free(dist_arr);
    return dist_arr;
}


/**
 * @todo 2. handle the situation when number of neighbors of source over 32
*/
void computeCC_shareBased(struct CSR* _csr, float* _CCs){
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

        //each neighbor mapping to bit_SI, if it haven't been source yet
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

int CC_CheckDistAns(struct CSR* _csr, float* _CCs, int _tempSourceID, int* dist){
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

void checkStackSequence(struct CSR* _csr, struct stack* _S_check, int* _dist_check, int _sourceID){
    // printf("Check stack sequence...\n");
    

    int* tempArr = (int*)malloc(sizeof(int) * _csr->csrVSize);
    memset(tempArr, 0, sizeof(int) * _csr->csrVSize);
    
    tempArr[_S_check->nodeIDs[0]]   = 1;
    tempArr[_sourceID]              = 1;
    for(int i = _S_check->top ; i > 0 ; i --){
        int node1 = _S_check->nodeIDs[i];
        if(tempArr[node1] == 1){
            printf("[Error] Stack Repeat push : S[%2d] = %2d\n", i, node1);
            exit(1);
        }
        if(_dist_check[_S_check->nodeIDs[i]] < _dist_check[_S_check->nodeIDs[i - 1]]){
            int node2 = _S_check->nodeIDs[i - 1];
            printf("[Error] Stack sequence : S[%2d] = %2d, dist = %2d, S[%2d] = %2d, dist = %2d\n", i, node1, _dist_check[node1], i - 1, node2, _dist_check[node2]);
            exit(1);
        }
        tempArr[node1] ++;
    }

    if((_S_check->top + 1) != (_csr->endNodeID - _csr->startNodeID + 1 - 1)){
        printf("[Error] Stack len = %2d, nodesNum = %2d\n", _S_check->top + 1, _csr->endNodeID - _csr->startNodeID);
        for(int i = _csr->startNodeID ; i <= _csr->endNodeID ; i ++){
            // if(tempArr[i] == _sourceID){continue;}

            if(tempArr[i] == 0){
                printf("\tlost node %2d, tempArr[%d] = %d\n", i, i, tempArr[i]);
                break;
            }
        }
        exit(1);
    }

    free(tempArr);
    printf("done\n");
}

int checkStackans2(struct stack* S_origin, struct stack* S_method1, struct CSR* _csr){
    // printf("\n\n");
    printf("Check Ans...%2d\n", nextBFS);
    int* existArr = calloc(_csr->csrVSize, sizeof(int));
    for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
        memset(existArr, 0, sizeof(int) * _csr->startNodeID);
        // printf("nodeID = %d : \n", nodeID);
        if(S_origin->predecessors[nodeID]->tail != S_method1->predecessors[nodeID]->tail){
            printf("predecessor[%d] : Ori.tail = %d, method1.tail = %d => len diff\n", nodeID, S_origin->predecessors[nodeID]->tail, S_method1->predecessors[nodeID]->tail);
            printf("ori[%d].pred     = {", nodeID);
            for(int predIndex = 0 ; predIndex <= S_origin->predecessors[nodeID]->tail ; predIndex ++){
                int predID = S_origin->predecessors[nodeID]->dataArr[predIndex];
                printf("%d, ", predID);
            }
            printf("}\n");
            printf("method1[%d].pred = {", nodeID);
            for(int predIndex = 0 ; predIndex <= S_method1->predecessors[nodeID]->tail ; predIndex ++){
                int predID = S_method1->predecessors[nodeID]->dataArr[predIndex];
                printf("%d, ", predID);
            }
            printf("}\n");
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
    // degreeOneFolding(csr);
    // showCSR(csr);
    float* BCs          = (float*)calloc(sizeof(float), csr->csrVSize);
    float* BCs2         = (float*)calloc(sizeof(float), csr->csrVSize);
    BCs_ori             = (float*)calloc(sizeof(float), csr->csrVSize);
    double time1        = 0;
    double time2        = 0;
    double BrandesTime  = 0;
    double CC_shareBasedTime = 0;
    double CC_ori       = 0;


    #ifdef Timing_Method1_And_Origin
    time1               = seconds();
    // offset              = 1;
    computeBC_shareBased(csr, BCs);

    time2               = seconds();
    method1_BC_time     = time2 - time1;
    
    time1               = seconds();
    computeBC(csr, BCs2);
    time2               = seconds();
    ori_BC_time         = time2 - time1;
    #endif

    #ifdef LiveJournal_Test
    computeBC_shareBased(csr, BCs);
    #endif
    // for(offset = csr->startNodeID ; offset <= csr->endNodeID ; offset ++){
    //     printf("SourceID = %2d!!!!!\n", offset);
    //     computeBC_shareBased2(csr, BCs);
    // }



    float* CCs          = (float*)calloc(sizeof(float), csr->csrVSize);
    
    // computeCC(csr, CCs);
    time1 = seconds();
    computeCC_shareBased(csr, CCs);
    time2 = seconds();
    CC_shareBasedTime = time2 - time1;

    
    time1 = seconds();
    computeCC(csr, CCs);
    time2 = seconds();
    CC_ori = time2 - time1;
    //CC 還沒記時間



    // BrandesTime         = method1_BC_time + ori_BC_time;
    // printf("[Execution Time] %2f (s)\n", BrandesTime);
    //紀錄時間
    FILE *fptr = fopen("CostTime.txt", "a");
    if(fptr == NULL){
        printf("[Error] OpenFile : Output.txt\n");
        exit(1);
    }
    
    // fprintf(fptr, "%s, method1, %f, ori, %f\n", datasetPath, method1_BC_time, ori_BC_time);
    fprintf(fptr, "%s, CC_shareBased %f, CC_ori %f\n", datasetPath, CC_shareBasedTime, CC_ori);
    // fprintf(fptr, "%s, %f\n", datasetPath, BrandesTime);
    fclose(fptr);
    //紀錄BC分數

    // fptr = fopen("BC_Score.txt", "a");
    // if(fptr == NULL){
    //     printf("[Error] OpenFile : Output.txt\n");
    //     exit(1);
    // }
    // fprintf(fptr, "%d %d\n", csr->startNodeID, csr->endNodeID);
    // for(int node = csr->startNodeID ; node <= csr->endNodeID ; node ++){
    //     fprintf(fptr, "%f\n", BCs[node]);
    // }
    // fclose(fptr);
}