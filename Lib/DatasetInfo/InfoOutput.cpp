#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#include "headers.h"

#include <map>
#include <iostream>
using namespace std;

int main(int argc, char* argv[]){
    char* datasetPath = argv[1];
    printf("exeName = %s\n", argv[0]);
    printf("datasetPath = %s\n", datasetPath);
    struct Graph* graph = buildGraph(datasetPath);
    struct CSR* csr     = createCSR(graph);

    map<int, int> eachDegreeCount;
    int degree = -1;
    for(int nodeID = csr->startNodeID ; nodeID <= csr->endNodeID ; nodeID++){
        degree = csr->csrNodesDegree[nodeID];
        if(eachDegreeCount.count(degree) > 0){
            eachDegreeCount[degree] ++;
        }
        else{
            eachDegreeCount[degree] = 1;
        }
    }
    
    for(int degree = 1 ; degree <= csr->maxDegree ; degree ++){
        cout << degree << " ";
        if(eachDegreeCount.count(degree) > 0){
            cout << eachDegreeCount[degree] << endl;
        }
        else{
            cout << 0 << endl;
        }
    }
}