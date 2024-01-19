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

#define Ordinary    1
#define D1          2
#define D1Hub       3

struct CSR{
    int* csrV;              //O(|V|)
    int* csrE;              //O(|E|)
    int csrVSize;           //結尾會多一個格子，放總共的edge數，當作最後的offset。
    int csrESize;           //如果是無向圖，csrESize就是原本dataset的兩倍。
    int* csrNodesDegree;    //O(|V|), 紀錄csr的各個node的degree。

    int* oriCsrV;           //O(|V|)
    int* oriCsrNodesDegree; //O(|V|), 紀錄原始csr的各個node的degree。
    struct qQueue* degreeOneNodesQ; //紀錄誰是degreeOne的Queue。
    int foldedDegreeOneCount;
    int startAtZero;
    int maxDegree;          //紀錄最大degree是多少

    int startNodeID;        //用於traverse
    int endNodeID;          //用於traverse
    int totalNodeNumber;    //用於traverse

    char* nodesType;        //用於紀錄所有node的類型
    int* representNode;     //用於紀錄有多少點被壓縮到該點

    float* BCs;
};

struct CSR* createCSR(struct Graph* _adjlist);
void swap(int* _val1, int* _val2);
void degreeOneFolding(struct CSR* _csr);
void showCSR(struct CSR* csr);
#endif