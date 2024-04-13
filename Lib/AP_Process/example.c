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
    double AP_Copy_And_Split_Time;
    time1 = seconds();
    D1Folding(csr);
    time2 = seconds();
    D1FoldingTime = time2 - time1;
    printf("[Execution Time] D1Folding          = %f\n", D1FoldingTime);

    time1 = seconds();
    AP_detection(csr);
    time2 = seconds();
    AP_detectionTime = time2 - time1;
    printf("[Execution Time] AP_detection       = %f\n", AP_detectionTime);

    // printf("csr->csrV[endNode] = %d\n", csr->csrV[csr->endNodeID]);
    // int nextCsrE_offset = csr->csrV[csr->endNodeID + 1];
    // printf("nextCsrE_offset = %d\n", nextCsrE_offset);
    // for(int i = nextCsrE_offset ; i < nextCsrE_offset + 10 ; i ++){
    //     csr->csrE[i] = i - nextCsrE_offset;
    //     printf("csrE[%d] = %d\n", i, csr->csrE[i]);

    // }
    time1 = seconds();
    AP_Copy_And_Split(csr);
    time2 = seconds();
    AP_Copy_And_Split_Time = time2 - time1;
    printf("[Execution Time] AP_Copy_And_Split  = %f\n", AP_Copy_And_Split_Time);
    printf("apCount = %d\n", csr->ap_count);
    printf("maxCompSize = %d\n", csr->maxCompSize_afterSplit);
}