#include "headers.h"
#include "AP_Process.h"
#include "AP_Process.c"

int main(int argc, char* argv[]){
    char* datasetPath = argv[1];
    printf("datasetPath = %s\n", datasetPath);
    struct Graph* adjList = buildGraph(datasetPath);
    struct CSR* csr = createCSR(adjList);
    // showCSR(csr);
    double time1, time2;
    double D1FoldingTime;
    double AP_detectionTime;

    time1 = seconds();
    D1Folding(csr);
    time2 = seconds();
    D1FoldingTime = time2 - time1;
    printf("[Execution Time] D1Folding      = %f\n", D1FoldingTime);

    time1 = seconds();
    AP_detection(csr);
    time2 = seconds();
    AP_detectionTime = time2 - time1;
    printf("[Execution Time] AP_detection   = %f\n", AP_detectionTime);

    AP_Copy_And_Split(csr);

}