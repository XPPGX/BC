/**
 * @author XPPGX
 * @date 2023/07/15
*/
#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

//content
#define CUDA

#ifndef ADJLIST
// #error Need include "AdjList.h", pls add "vVector.h" into "headers.h"
#include "../AdjList/AdjList.h"
#endif

#ifndef QQueue
// #error Need include "qQueue.h", pls add "qQueue.h" into "headers.h"
#include "../qQueue/qQueue.h"
#endif

#ifndef cCSR
#define cCSR

#define Ordinary    0x00000001U
#define D1          0x00000002U
#define D1Hub       0x00000004U
#define OriginAP    0x00000008U
#define ClonedAP    0x00000010U

struct CSR{
    int* csrV;              //O(2 * |V|)
    int* csrE;              //O(4 * |E|)
    int csrVSize;           //結尾會多一個格子，放總共的edge數，當作最後的offset
    int csrESize;           //如果是無向圖，csrESize就是原本dataset的兩倍
    int* csrNodesDegree;    //O(|V|), 紀錄csr的各個node的degree

    int* oriCsrV;           //O(|V|)
    int* oriCsrNodesDegree; //O(|V|), 紀錄原始csr的各個node的degree

    unsigned int* nodesType;//用於紀錄所有node的類型

    int* representNode;     //用於紀錄有多少點被壓縮到該點
    int* ff;                //用於紀錄壓縮進自己node的dist是多少, 參考自BADIOS論文(Graph manipulation)

    struct qQueue* degreeOneNodesQ; //紀錄誰是degreeOne的Queue
    int* notD1Node;         //用於紀錄非degreeOne的Node
    int* D1Parent;          //紀錄每個D1，最靠近component的祖先是誰
    int foldedDegreeOneCount;
    int hubNodeCount;
    int ordinaryNodeCount;
    
    int startAtZero;
    int maxDegree;          //紀錄最大degree是多少

    int startNodeID;        //用於traverse
    int endNodeID;          //用於traverse
    int totalNodeNumber;    //用於traverse

    int* low;               //偵測AP用的
    int* depth_level;       //偵測AP用的
    int* Dfs_sequence;      //紀錄哪個 node 先被 DFS 探到
    int ap_count;           //紀錄有多少個 AP
    int* AP_List;           //儲存所有 AP nodeID
    int* AP_component_number;  //紀錄 該 AP 連接多少個component //可能無用
    int* compID;            //儲存每個node在哪個component
    int* apCloneTrackOriAp_ID;   //紀錄 AP分身 原本的 AP本尊是誰
    int apCloneCount;       //紀錄創建了幾個 AP分身
    int compNum;            //紀錄comp有幾個
    int maxCompSize_afterSplit;        //紀錄AP切割後，最大的comp有幾個點

    int* aliveNode;         //紀錄 "非D1", "非 ori AP", "非 AP clone" 的nodes
    int aliveNodeCount;     //紀錄有幾個aliveNode
    int* aliveNodeFlags;    //紀錄每個node是否還需要當source，如果需要，則flag = 1

    int* comp_NodesID_CsrData;  //把aliveNode中，相同comp的nodes放在一起
    int* comp_CsrOffset;        //csr offset for nodeID with same component
    int* nodesCompID;           //nodesCompID 把每個 aliveNodes 都 assign compID

    int* mapNodeID_New_to_Old; //每個nodeID對應到的新nodeID，index是newID，value是oldID
    int* mapNodeID_Old_to_new; //每個nodeID對應到的舊nodeID，index是oldID，value是newID
    int* orderedCsrV; //O(2 * |V|)
    int* orderedCsrE; //O(4 * |V|)
    int newEndID;
    int* comp_newNodesID_CsrData;  //把aliveNode中，相同comp的nodes放在一起
    int* comp_newCsrOffset;        //csr offset for nodeID with same component
    int* newNodesCompID;           //nodesCompID 把每個 aliveNodes 都 assign compID (aliveNode 代表 還存在在graph中的點，(degree > 0的點))
    int compEndID;              //紀錄最後一個compID
    
    float* BCs;
    int* CCs;
};

struct CSR* createCSR(struct Graph* _adjlist);
void swap(int* _val1, int* _val2);
void showCSR(struct CSR* csr);
#endif