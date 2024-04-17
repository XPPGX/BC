#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif


#ifndef ADJLIST
// #error Need include "AdjList.h", pls add "vVector.h" into "headers.h"
#include "../AdjList/AdjList.h"
#include "../AdjList/AdjList.c"
#endif

#ifndef QQueue
// #error Need include "qQueue.h", pls add "qQueue.h" into "headers.h"
#include "../qQueue/qQueue.h"
#include "../qQueue/qQueue.c"
#endif

#ifndef cCSR
// #error Need include "CSR.h", pls add "CSR.h" into "headers.h"
#include "../CSR/CSR.h"
#include "../CSR/CSR.c"
#endif

#ifndef AP_Process
#define AP_Process

struct part_info{
    int apID;
    int compID;
    int represent;
    int ff;
};

struct apClone_info{
    int* Ori_apNodeID;
    int* apCloneID;

    int* apCloneff;
    int* apCloneRepresent;

    int* apCloneCsrV;
    // int* apCloneCsrE;
    
    int newNodeCount;
    int apCloneCsrV_offsetCounter;
    int apCloneCsrE_offsetCounter;
};

/**
 * @brief
 * Find all APs existed in current graph, record them into _csr->AP_List
 * This function can be called after D1Folding or before D1Folding
*/
void AP_detection(struct CSR* _csr);


void quicksort_nodeID_with_data(int* _nodes, int* _data, int _left, int _right);
/**
 * @brief
 * If an AP u has AP neighbor v, then (u, v) is a bridge 
 * 1. 依照 apNum 由小到大排序 _csr->AP_List 中的 AP
 * 2. 從 apNum 大的開始進行 split
*/
void AP_Copy_And_Split(struct CSR* _csr);


#endif