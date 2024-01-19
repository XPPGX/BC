#include "headers.h"
#include "D1Process.h"
#include "D1Process.c"

int main(int argc, char* argv[]){
    char* datasetPath = argv[1];
    printf("datasetPath = %s\n", datasetPath);
    struct Graph* adjList = buildGraph(datasetPath);
    struct CSR* csr = createCSR(adjList);
    // showCSR(csr);

    // struct VirtualNodes* SV = init_SV();
    float* BCs = (float*)calloc(sizeof(float), csr->csrVSize);
    D1Folding(csr);
    /*
    for(int i = 0 ; i < csr->csrVSize ; i ++){
        switch(csr->nodesType[i]){
            case D1:
                printf("D1\t%d\n", i);
                break;
            case D1Hub:
                printf("D1Hub\t%d\n", i);
                break;
        }
    }
    */
    printf("d1Node count = %d\n", csr->foldedDegreeOneCount);
}