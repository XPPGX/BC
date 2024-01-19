/**
 * @author XPPGX
 * @date 2023/07/15
*/

#include "AdjList.h"

//default the graph is undirected graph

// #define _DEBUG_
#define rowSize 2000

struct Graph* buildGraph(char* _datasetPath){
    struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
    FILE* fptr = fopen(_datasetPath, "r");
    if(fptr == NULL){
        printf("Error: Open file\n");
        exit(1);
    }
    char row[rowSize];
    int val1, val2 = 0;
    fgets(row, rowSize, fptr);
    
    getRowData(row, &val1, &val2);
    
    #ifdef _DEBUG_
    printf("nodeNum = %d, edgeNum = %d\n", val1, val2);
    #endif

    graph->nodeNum = val1;
    graph->edgeNum = val2;
    
    //read first edge to check 0 or 1 is start index.
    fgets(row, rowSize, fptr);
    getRowData(row, &val1, &val2);
    if(val1 == 0 || val2 == 0){
        #ifdef _DEBUG_
        printf("Graph start with 0\n");
        #endif

        graph->startAtZero = 1;
        graph->vertices = (struct adjList*)malloc(sizeof(struct adjList) * graph->nodeNum);
        for(int i = 0 ; i < graph->nodeNum ; i ++){
            graph->vertices[i].neighbors = InitvVector();
        }
        // 宣告nodeDegree array的長度，因為nodeID從0開始，所以長度是nodeNum
        graph->nodeDegrees = (int*)calloc(graph->nodeNum, sizeof(int));
    }
    else{
        #ifdef _DEBUG_
        printf("Graph start with 1\n");
        #endif

        graph->startAtZero = 0;
        graph->vertices = (struct adjList*)malloc(sizeof(struct adjList) * (graph->nodeNum + 1));
        for(int i = 0 ; i < graph->nodeNum + 1 ; i ++){
            graph->vertices[i].neighbors = InitvVector();
        }
        // 宣告nodeDegree array的長度，因為nodeID從1開始，所以長度是nodeNum + 1;
        graph->nodeDegrees = (int*)calloc(graph->nodeNum + 1, sizeof(int));
    }

    //append first edge into adjlist
    vAppend(graph->vertices[val1].neighbors, val2);
    graph->nodeDegrees[val1] ++;
    vAppend(graph->vertices[val2].neighbors, val1);
    graph->nodeDegrees[val2] ++;
    //run over all remain edges
    for(int i = 1 ; i < graph->edgeNum ; i ++){
        fgets(row, rowSize, fptr);
        getRowData(row, &val1, &val2);
        vAppend(graph->vertices[val1].neighbors, val2);
        graph->nodeDegrees[val1] ++;
        vAppend(graph->vertices[val2].neighbors, val1);
        graph->nodeDegrees[val2] ++;
    }
    
    //初始化degreeOneQueue
    graph->degreeOneQueue = InitqQueue();

    //對每個node的neighbors根據degree進行排列 and 收集degreeOne
    if(graph->startAtZero == 1){ //nodeID從0開始
        for(int nodeID = 0 ; nodeID < graph->nodeNum ; nodeID ++){
            struct vVector* neighbors = graph->vertices[nodeID].neighbors;
            int left = 0;
            int right = graph->vertices[nodeID].neighbors->tail;

            // 如果nodeID的degree == 1，則放入degreeOneQueue，否則根據degree對nodeID的neighbor進行排序
            if(graph->nodeDegrees[nodeID] == 1){
                qPushBack(graph->degreeOneQueue, nodeID);
            }
            else{
                quicksort(neighbors, graph->nodeDegrees, left, right);
            }
        }
    }
    else{ //nodeID從1開始
        for(int nodeID = 1 ; nodeID < graph->nodeNum + 1 ; nodeID ++){
            #ifdef _DEBUG_
            printf("==========================\n");
            printf("nodeID = %d\n", nodeID);
            #endif
            struct vVector* neighbors = graph->vertices[nodeID].neighbors;
            int left = 0;
            int right = graph->vertices[nodeID].neighbors->tail;

            // 如果nodeID的degree == 1，則放入degreeOneQueue，否則根據degree對nodeID的neighbor進行排序
            if(graph->nodeDegrees[nodeID] == 1){
                qPushBack(graph->degreeOneQueue, nodeID);
            }
            else{
                quicksort(neighbors, graph->nodeDegrees, left, right);
            }
        }
    }

    printf("Success : Build AdjList\n");
    printf("==============================\n");
    printf("dataset = %s\n", _datasetPath);
    printf("nodeNum = %d\n", graph->nodeNum);
    printf("edgeNum = %d\n", graph->edgeNum);
    printf("degreeOneNum = %d\n", graph->degreeOneQueue->rear + 1);
    switch(graph->startAtZero){
        case 0:
            printf("graph start with 1\n");
            break;
        case 1:
            printf("graph start with 0\n");
            break;
    }
    printf("==============================\n");
    fclose(fptr);
    return graph;
}

void quicksort(struct vVector* _neighbors, int* _nodeDegrees, int _left, int _right){
    if(_left > _right){
        return;
    }
    int smallerAgent = _left;
    int smallerAgentNode = -1;
    int equalAgent = _left;
    int equalAgentNode = -1;
    int largerAgent = _right;
    int largerAgentNode = -1;

    int pivotNode = _neighbors->dataArr[_right];

    int tempNode = 0;

    while(equalAgent <= largerAgent){
        #ifdef _DEBUG_
        printf("smallerAgent = %d, equalAgent = %d, largerAgent = %d\n", smallerAgent, equalAgent, largerAgent);
        #endif

        smallerAgentNode = _neighbors->dataArr[smallerAgent];
        equalAgentNode = _neighbors->dataArr[equalAgent];
        largerAgentNode = _neighbors->dataArr[largerAgent];
        
        if(_nodeDegrees[equalAgentNode] < _nodeDegrees[pivotNode]){ //equalAgentNode的degree < pivotNode的degree
            // swap smallerAgentNode and equalAgentNode
            tempNode = _neighbors->dataArr[smallerAgent];
            _neighbors->dataArr[smallerAgent] = _neighbors->dataArr[equalAgent];
            _neighbors->dataArr[equalAgent] = tempNode;

            smallerAgent ++;
            equalAgent ++;
        }
        else if(_nodeDegrees[equalAgentNode] > _nodeDegrees[pivotNode]){ //equalAgentNode的degree > pivotNode的degree
            // swap largerAgentNode and equalAgentNode
            tempNode = _neighbors->dataArr[largerAgent];
            _neighbors->dataArr[largerAgent] = _neighbors->dataArr[equalAgent];
            _neighbors->dataArr[equalAgent] = tempNode;

            largerAgent --;
        }
        else{ //equalAgentNode的degree == pivotNode的degree
            equalAgent ++;
        }
    }
    
    #ifdef _DEBUG_
    printf("[Finished] smallerAgent = %d, equalAgent = %d, largerAgent = %d\n", smallerAgent, equalAgent, largerAgent);
    printf("after partition:\n");
    printf("arr = {");
    for(int neighborIndex = 0 ; neighborIndex < _neighbors->tail + 1 ; neighborIndex ++){
        printf("%d(%d), ", _neighbors->dataArr[neighborIndex], _nodeDegrees[_neighbors->dataArr[neighborIndex]]);
    }
    printf("}\n");
    #endif

    // smallerAgent現在是pivot key的開頭
    // largerAgent現在是pivotKey的結尾
    quicksort(_neighbors, _nodeDegrees, _left, smallerAgent - 1);
    quicksort(_neighbors, _nodeDegrees, largerAgent + 1, _right);
}

void showAdjList(struct Graph* _graph){
    if(_graph->startAtZero == 1){
        printf("Graph start at zero\n");
        for(int i = 0 ; i < _graph->nodeNum ; i ++){
            printf("neighbor[%d] = {", i);
            for(int neighbor_index = 0 ; neighbor_index <= _graph->vertices[i].neighbors->tail ; neighbor_index++){
                printf("%d, ", _graph->vertices[i].neighbors->dataArr[neighbor_index]);
            }
            printf("}\n");
        }
    }
    else{
        printf("Graph start at one\n");
        for(int i = 1 ; i < _graph->nodeNum + 1 ; i ++){
            printf("neighbor[%d] = {", i);
            for(int neighbor_index = 0 ; neighbor_index <= _graph->vertices[i].neighbors->tail ; neighbor_index++){
                printf("%d, ", _graph->vertices[i].neighbors->dataArr[neighbor_index]);
            }
            printf("}\n");
        }
    }

    printf("Degree one queue : {");
    while(!qIsEmpty(_graph->degreeOneQueue)){
        printf("%d, ", qPopFront(_graph->degreeOneQueue));
    }
    printf("}\n");
}