#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#include <iostream>
#include "headers.h"
#include <map>
using namespace std;

int main(int argc, char* argv[]){
    char* datasetPath = argv[1];
    printf("exeName = %s\n", argv[0]);
    printf("datasetPath = %s\n", datasetPath);
    struct Graph* graph = buildGraph(datasetPath);
    struct CSR* csr     = createCSR(graph);

    map<int, int> degree_map;
    for(int nodeID = csr->startNodeID ; nodeID <= csr->endNodeID ; nodeID ++){
        int degree = csr->csrNodesDegree[nodeID];
        if(degree_map.count(degree) > 0){
            degree_map[degree] ++;
        }
        else{
            degree_map[degree] = 1;
        }
    }

    for(const auto& pair : degree_map){
        cout << pair.first << "," << pair.second << endl;
    }
}