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
#ifndef VVECTOR
// #error Need include "vVector.h", pls add "vVector.h" into "headers.h"
#include "../vVector/vVector.h"
#endif 

#ifndef FileReader
// #error Need include "FileReader.h", pls add "FileReader.h" into "headers.h"
#include "../FileReader/FileReader.h"
#endif

#ifndef QQueue
// #error Need include "qQueue.h", pls add "qQueue.h" into "headers.h"
#include "../qQueue/qQueue.h"
#endif

#ifndef ADJLIST
#define ADJLIST
struct adjList{
    struct vVector* neighbors;
};

struct Graph{
    int nodeNum;
    int edgeNum;
    int startAtZero;
    int* nodeDegrees;
    struct adjList* vertices;
    struct qQueue* degreeOneQueue;
};

struct Graph* buildGraph(char* _datasetPath);
void quicksort(struct vVector* _neighbors, int* _nodeDegrees, int _left, int _right);
void showAdjList(struct Graph* _graph);
#endif