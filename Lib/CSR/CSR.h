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
#error Need include "AdjList.h", pls add "vVector.h" into "headers.h"
#endif

#ifndef QQueue
#error Need include "qQueue.h", pls add "qQueue.h" into "headers.h"
#endif

#ifndef cCSR
#define cCSR

#define Ordinary    0x01
#define D1          0x02
#define D1Hub       0x04
#define OriginAP    0x08
#define ClonedAP    0x10

struct CSR{
    int* csrV;              //O(2 * |V|)
    int* csrE;              //O(4 * |E|)
    int csrVSize;           //結尾會多一個格子，放總共的edge數，當作最後的offset
    int csrESize;           //如果是無向圖，csrESize就是原本dataset的兩倍
    int* csrNodesDegree;    //O(|V|), 紀錄csr的各個node的degree

    int* oriCsrV;           //O(|V|)
    int* oriCsrNodesDegree; //O(|V|), 紀錄原始csr的各個node的degree

    char* nodesType;        //用於紀錄所有node的類型

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

    int ap_count;           //紀錄有多少個 AP
    int* AP_List;           //儲存所有 AP nodeID
    int* AP_component_number;  //紀錄 該 AP 連接多少個component //可能無用
    int* compID;            //儲存每個node在哪個component
    int* apCloneTrackOriAp_ID;   //紀錄 AP分身 原本的 AP本尊是誰
    int apCloneCount;       //紀錄創建了幾個 AP分身
    int compNum;            //紀錄comp有幾個
    int maxCompSize_afterSplit;        //紀錄AP切割後，最大的comp有幾個點

    float* BCs;
    int* CCs;
};

struct CSR* createCSR(struct Graph* _adjlist);
void swap(int* _val1, int* _val2);
void showCSR(struct CSR* csr);
#endif