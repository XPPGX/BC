/**
 * @author XPPGX
 * @date 2023/07/15
*/
#include "CSR.h"

//#define _DEBUG_

struct CSR* createCSR(struct Graph* _adjlist){
    printf("==============================\n");
    printf("CreateCSR...\n");
    int* CSR_V;
    int* CSR_E;
    int nodeID = 0;
    int tempNodeNum = 0;
    if(_adjlist->startAtZero == 1){
        // printf("Graph startAtZero = yes\n");
        tempNodeNum = _adjlist->nodeNum + 1;
        CSR_V = (int*)malloc(sizeof(int) * tempNodeNum);
        nodeID = 0;
    }
    else{
        // printf("Graph startAtZero = no\n");
        tempNodeNum = _adjlist->nodeNum + 2;
        CSR_V = (int*)malloc(sizeof(int) * tempNodeNum);
        CSR_V[0] = -1;
        nodeID = 1;
    }
    CSR_E = (int*)malloc(sizeof(int) * _adjlist->edgeNum * 2);
    int indexCount = -1;

    int maxDegree = 0;
    // printf("nodeID = %d, tempNodeNum = %d, indexCount = %d\n", nodeID, tempNodeNum, indexCount);
    for(; nodeID < tempNodeNum - 1 ; nodeID ++){
        CSR_V[nodeID] = indexCount + 1;

        #ifdef _DEBUG_
        printf("CSR_V[%d] = %d\n", nodeID, CSR_V[nodeID]);
        #endif
        int neighborIndex;
        for(neighborIndex = 0 ; neighborIndex <= _adjlist->vertices[nodeID].neighbors->tail ; neighborIndex ++){
            indexCount ++;
            CSR_E[indexCount] = _adjlist->vertices[nodeID].neighbors->dataArr[neighborIndex];
        }
        if(maxDegree < neighborIndex){
            maxDegree = neighborIndex;
        }
    }
    // printf("nodeID = %d, tempNodeNum = %d, indexCount = %d\n", nodeID, tempNodeNum, indexCount);
    CSR_V[nodeID] = _adjlist->edgeNum * 2;
    //對csr結構賦值
    struct CSR* csr = (struct CSR*)malloc(sizeof(struct CSR));
    csr->csrV = CSR_V;

    

    csr->csrE               = CSR_E;
    csr->csrVSize           = tempNodeNum;
    csr->csrESize           = _adjlist->edgeNum * 2;
    csr->csrNodesDegree     = _adjlist->nodeDegrees;
    csr->maxDegree          = maxDegree;

    csr->oriCsrNodesDegree  = (int*)malloc(sizeof(int) * csr->csrVSize);
    memcpy(csr->oriCsrNodesDegree, csr->csrNodesDegree, sizeof(int) * csr->csrVSize);

    csr->oriCsrV            = (int*)malloc(sizeof(int) * csr->csrVSize);
    memcpy(csr->oriCsrV, csr->csrV, sizeof(int) * csr->csrVSize);

    csr->nodesType          = (char*)malloc(sizeof(char) * csr->csrVSize);
    memset(csr->nodesType, 1, sizeof(char) * csr->csrVSize);

    csr->representNode      = (int*)malloc(sizeof(int) * csr->csrVSize);
    for(int i = 0 ; i < csr->csrVSize ; i ++){
        csr->representNode[i] = 1;
    }

    csr->BCs                = (float*)calloc(sizeof(float), csr->csrVSize);

    csr->startAtZero        = _adjlist->startAtZero;


    if(csr->startAtZero){
        csr->totalNodeNumber = csr->csrVSize - 1;
    }
    else{
        csr->totalNodeNumber = csr->csrVSize - 2;
    }
    csr->startNodeID    = !(csr->startAtZero);
    csr->endNodeID      = csr->totalNodeNumber - csr->startAtZero;
    

    
    csr->degreeOneNodesQ = _adjlist->degreeOneQueue;
    csr->foldedDegreeOneCount = 0;
    printf("csr->csrVSize = %d\n", csr->csrVSize);
    printf("csr->csrESize = %d\n", csr->csrESize);
    printf("csr->degreeOneNum = %d\n", csr->degreeOneNodesQ->rear + 1);
    printf("csr->startAtZero = %d\n", csr->startAtZero);
    printf("[Success] CreateCSR finish\n");
    printf("==============================\n");
    return csr;
}

void swap(int* _val1, int* _val2){
    int temp = *_val1;
    *_val1 = *_val2;
    *_val2 = temp;
}

void degreeOneFolding(struct CSR* _csr){
    struct qQueue* d1Q = _csr->degreeOneNodesQ;

    int d1Node = -1;

    int hubNode = -1;
    int currentNeighbor = -1;

    while(!qIsEmpty(d1Q)){
        d1Node = qPopFront(d1Q);
        hubNode = _csr->csrE[_csr->csrV[d1Node]];
        
        #ifdef _DEBUG_
        printf("%d, linking to %d\n", d1Node, hubNode);
        #endif
        // 找hubNode的neighbor中，d1Node所在的index，並把d1Node跟hubNode的當前第一個鄰居交換。
        // hubNode的offset + 1，因為hubNode拔掉一個neighbor。
        for(int hubNodeNeighborIndex = 0 ; hubNodeNeighborIndex < _csr->csrNodesDegree[hubNode] ; hubNodeNeighborIndex ++){
            currentNeighbor = _csr->csrE[_csr->csrV[hubNode] + hubNodeNeighborIndex];
            if(currentNeighbor == d1Node){
                swap(&(_csr->csrE[_csr->csrV[hubNode] + hubNodeNeighborIndex]), &(_csr->csrE[_csr->csrV[hubNode]]));
                break;
            }
        }
        // d1Node的offset不用 + 1，因為d1Node之後還要藉由hubNode去塗色
        _csr->csrV[d1Node] ++;
        // d1Node的degree - 1
        _csr->csrNodesDegree[d1Node] --;

        // hubNode的offset + 1
        _csr->csrV[hubNode] ++;
        // hubNode的degree - 1
        _csr->csrNodesDegree[hubNode] --;
        // 計數有多少degreeOne
        _csr->foldedDegreeOneCount ++;


        // 如果d1Node需要找 hubNode，則
        // _csr->csrV[d1Node] --; 的索引位置就是hubNode
    }

    #ifdef _DEBUG_
    printf("Folded Degree One Count = %d\n", _csr->foldedDegreeOneCount);
    #endif
}

void showCSR(struct CSR* csr){
    printf("show CSR...\n");
    int nodeID = 0;
    if(csr->startAtZero == 0){
        nodeID ++;
    }
    int neighborNum = 0;
    for(; nodeID < csr->csrVSize - 1 ; nodeID ++){
        neighborNum = csr->csrV[nodeID + 1] - csr->csrV[nodeID];
        printf("neighborNum = %d, neighbor[%d] = {", neighborNum, nodeID);
        for(int i = 0 ; i < neighborNum ; i ++){
            printf("%d, ", csr->csrE[csr->csrV[nodeID] + i]);
        }
        printf("}\n");
    }
    printf("Success : show CSR Finish.\n");
}