#include "headers.h"
#include "AP_Process.h"
#include "AP_Process.c"


int checkCC_Ans(struct CSR* _csr, int checkNodeID){
    struct qQueue* Q = InitqQueue();
    qInitResize(Q, _csr->csrVSize);
    int* dist_arr = (int*)malloc(sizeof(int) * _csr->csrVSize);
    memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);

    dist_arr[checkNodeID] = 0;
    qPushBack(Q, checkNodeID);

    int CC_ans = 0;

    while(!qIsEmpty(Q)){
        int curID = qPopFront(Q);
        // printf("curID = %d\n", curID);

        for(int nidx = _csr->csrV[curID] ; nidx < _csr->csrV[curID + 1] ; nidx ++){
            int nid = _csr->csrE[nidx];
            
            if(dist_arr[nid] == -1){
                qPushBack(Q, nid);
                dist_arr[nid] = dist_arr[curID] + 1;
                CC_ans += dist_arr[nid];
            }
        }
    }

    free(Q->dataArr);
    free(Q);
    free(dist_arr);

    return CC_ans;
}

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

    // int checkNodeID = 30;
    // int checkNode_CC_ans = checkCC_Ans(csr, checkNodeID);
    // int AP = 19222;
    // int nAP = 22635;
    // for(int nidx = csr->csrV[AP] ; nidx < csr->csrV[AP + 1 ] ; nidx ++){
    //     int nid = csr->csrE[nidx];
    //     if(nid == nAP){
    //         printf("(%d, %d) exist\n", AP, nAP);
    //     }
    // }

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

    
    time1 = seconds();
    AP_Copy_And_Split(csr);
    time2 = seconds();
    AP_Copy_And_Split_Time = time2 - time1;
    printf("[Execution Time] AP_Copy_And_Split  = %f\n", AP_Copy_And_Split_Time);
    printf("apCount = %d\n", csr->ap_count);
    printf("maxCompSize = %d\n", csr->maxCompSize_afterSplit);

    
    // printf("\n\n");
    // printf("[Dist Ans Checking]...\n");
    // printf("checkNodeID = %d, compID[%d] = %d\n", checkNodeID, checkNodeID, csr->compID[checkNodeID]);
    // printf("csrVSize + apCloneCount = %d\n", csr->csrVSize + csr->apCloneCount);
    // struct qQueue* Q = InitqQueue();
    // qInitResize(Q, csr->csrVSize + csr->apCloneCount);
    // int* dist_arr = (int*)malloc(sizeof(int) * (csr->csrVSize + csr->apCloneCount));
    // memset(dist_arr, -1, sizeof(int) * (csr->csrVSize + csr->apCloneCount));

    // dist_arr[checkNodeID] = 0;
    // qPushBack(Q, checkNodeID);
    // csr->CCs[checkNodeID] = 0;
    // while(!qIsEmpty(Q)){
    //     int currentNodeID = qPopFront(Q);

    //     // printf("curID = %d : {", currentNodeID);
    //     for(int nidx = csr->csrV[currentNodeID] ; nidx < csr->oriCsrV[currentNodeID + 1] ; nidx ++){
    //         int nid = csr->csrE[nidx];

    //         if(dist_arr[nid] == -1){
    //             qPushBack(Q, nid);
    //             dist_arr[nid] = dist_arr[currentNodeID] + 1;
    //             // printf("%d(%d), ", nid, dist_arr[nid]);
    //             if(csr->ff[nid] > 100000){
    //                 printf("(%d, %d)! ff[%d] = %d\n", currentNodeID, nid, nid, csr->ff[nid]);
    //             }
    //             csr->CCs[checkNodeID] += csr->ff[nid] + dist_arr[nid] * csr->representNode[nid];
    //         }
    //     }
    //     // printf("}\n");
    //     // break;
    // }
    // csr->CCs[checkNodeID] += csr->ff[checkNodeID];
    // printf("\nCC[%d] = %d, true_ans = %d\n", checkNodeID, csr->CCs[checkNodeID], checkNode_CC_ans);
}