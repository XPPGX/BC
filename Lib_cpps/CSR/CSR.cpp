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
        CSR_V = (int*)malloc(sizeof(int) * (2 * tempNodeNum));
        nodeID = 0;
    }
    else{
        // printf("Graph startAtZero = no\n");
        tempNodeNum = _adjlist->nodeNum + 2;
        CSR_V = (int*)malloc(sizeof(int) * (2 * tempNodeNum));
        CSR_V[0] = -1;
        nodeID = 1;
    }

    /**
     * CSR_E : 4 times of _adjlist->edgeNum
     * first "2" means undirected edges need to be record two times
     * second "2" is the space that preserving for new node such as : AP clone...
    */
    CSR_E = (int*)malloc(sizeof(int) * (_adjlist->edgeNum) * 2 * 2); 
    
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

    csr->oriCsrV            = (int*)malloc(sizeof(int) * (csr->csrVSize) * 2);
    memcpy(csr->oriCsrV, csr->csrV, sizeof(int) * (csr->csrVSize) * 2);

    csr->nodesType          = (unsigned int*)malloc(sizeof(unsigned int) * (csr->csrVSize) * 2);
    memset(csr->nodesType, 0, sizeof(unsigned int) * (csr->csrVSize) * 2);

    csr->CCs                = (int*)malloc(sizeof(int) * (csr->csrVSize) * 2);
    memset(csr->CCs, 0, sizeof(int) * (csr->csrVSize) * 2);
    // csr->BCs                = (float*)calloc(sizeof(float), csr->csrVSize);
    

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
    printf("csr->maxDegree = %d\n", csr->maxDegree);
    printf("[Success] CreateCSR finish\n");
    printf("==============================\n");
    return csr;
}

void swap(int* _val1, int* _val2){
    int temp = *_val1;
    *_val1 = *_val2;
    *_val2 = temp;
}

void showCSR(struct CSR* csr){
    printf("show CSR...\n");
    int nodeID = 0;
    if(csr->startAtZero == 0){
        nodeID ++;
    }
    int neighborNum = 0;
    for(nodeID = csr->startNodeID ; nodeID < csr->endNodeID ; nodeID ++){
        neighborNum = csr->csrV[nodeID + 1] - csr->csrV[nodeID];
        printf("neighborNum = %d, neighbor[%d] = {", neighborNum, nodeID);
        for(int i = 0 ; i < neighborNum ; i ++){
            printf("%d, ", csr->csrE[csr->csrV[nodeID] + i]);
        }
        printf("}\n");
    }
    printf("Success : show CSR Finish.\n");
}