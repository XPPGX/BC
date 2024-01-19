#ifndef COMMON
#define COMMON

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#endif

#include "headers.h"
#include "AdjList.h"
#include "AdjList.c"

int main(int argc, char* argv[]){
    char* datasetPath = argv[1];
    struct Graph* graph = buildGraph(datasetPath);
    
    // showAdjList(graph);
}