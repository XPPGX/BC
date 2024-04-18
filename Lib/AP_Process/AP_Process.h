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
    // int apID;
    // int compID;
    int partID;
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

//取得所有 nodes 對 _apNodeID 而言 是在哪個 part，assign part id 到 _partID 的 array中
void assignPartId_for_AP(struct CSR* _csr, int* _partID, int _apNodeID, struct qQueue* _Q);

//取得 _apNodeID 的 neighbors中 (是AP) && (該AP 跟 _apNodeID 共享某個 part)
void findInterfaceAPs(struct CSR* _csr, int* _partID,  int* _eachPartNeighborNum, int* _partInterfaceAP, int _apNodeID);

//取得每個 part 的 w 跟 ff
int getPartsInfo(struct CSR* _csr, int* _partID, int _apNodeID, struct qQueue* _Q, struct part_info* _parts, int _maxBranch, int* _partFlag, int* _dist_arr, int* _total_represent, int* _total_ff);


/**
 * @brief
 * If an AP u has AP neighbor v, then (u, v) is a bridge 
 * 1. 依照 apNum 由小到大排序 _csr->AP_List 中的 AP
 * 2. 從 apNum 大的開始進行 split
*/
void AP_Copy_And_Split(struct CSR* _csr);


#endif