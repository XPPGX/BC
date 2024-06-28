#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif


#ifndef ADJLIST
// #error Need include "AdjList.h", pls add "vVector.h" into "headers.h"
#include "../AdjList/AdjList.h"
#endif

#ifndef QQueue
// #error Need include "qQueue.h", pls add "qQueue.h" into "headers.h"
#include "../qQueue/qQueue.h"
#include "../qQueue/qQueue.cpp"
#endif

#ifndef cCSR
// #error Need include "CSR.h", pls add "CSR.h" into "headers.h"
#include "../CSR/CSR.h"
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

struct newID_info{
    int ff;
    int w;
};
/**
 * @brief
 * Find all APs existed in current graph, record them into _csr->AP_List
 * This function can be called after D1Folding or before D1Folding
 * 
 * It's just Tarjan' AP detection algorithm
*/
void AP_detection(struct CSR* _csr);


void quicksort_nodeID_with_data(int* _nodes, int* _data, int _left, int _right);

//取得所有 nodes 對 _apNodeID 而言 是在哪個 part，assign part id 到 _partID 的 array中
/**
 * @brief
 * 取得所有 nodes 對 _apNodeID 而言 是在哪個 part，assign part id 到 _partID 的 array中
 * @return 有幾個 part
*/
int assignPartId_for_AP(struct CSR* _csr, int* _partID, int _apNodeID, struct qQueue* _Q);

//取得 _apNodeID 周圍的 ap neighbor 是否是 part interface
/**
 * @brief
 * 取得 _apNodeID 周圍的 ap neighbor 是否是 part interface，記錄在 _partInterfaceAP
 * @param _partID 記錄對 _apNodeID 而言，所有的 node 的 partID (index = nodeID, value = partID)
 * @param _eachPartNeighborNum 紀錄對 _apNodeID 而言，每個 part 的 neighbor 各有幾個 (index = partID, value = neighborNum)
 * @param _partInterfaceAP 紀錄對 _apNodeID 而言，每個 part 的連接點是否是 AP，如果是則記錄該 AP 的 nodeID (index = partID, value = apID)
 * @param _apNodeID 當前所關注的 apNode
 * @param _apCloneFlag 用於判斷是否要為這個 AP本尊 創建 AP分身，create apClone if _apCloneFlag == 1, else not.
*/
void findInterfaceAPs(struct CSR* _csr, int* _partID,  int* _eachPartNeighborNum, int* _partInterfaceAP, int _apNodeID);

//取得每個 part 的 w 跟 ff
int getPartsInfo(struct CSR* _csr, int* _partID, int _apNodeID, struct qQueue* _Q, struct part_info* _parts, int _maxBranch, int* _partFlag, int* _dist_arr, int* _total_represent, int* _total_ff);


/**
 * @brief
 * 判斷 AP本尊 是否需要創建 AP分身，並切割 graph
*/
void AP_Copy_And_Split(struct CSR* _csr);

//把相同component的nodeID 放在一起(只包含aliveNode)
void compactNodesByComp(struct CSR* _csr);

/**
 * 把相同component的 csrV, csrE放在一起(不包含 "D1", "已被切光edge的originAP")
 * assign oldID.ff, oldID.w to newID_infos for each newID
 * 
 * rebuild graph for better memory access
*/
struct newID_info* rebuildGraph(struct CSR* _csr);

#endif