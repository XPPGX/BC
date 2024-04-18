#include "headers.h"
#include "AP_Process.h"
#include "AP_Process.c"

// #define AfterD1Folding
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

        for(int nidx = _csr->csrV[curID] ; nidx < _csr->oriCsrV[curID + 1] ; nidx ++){
            int nid = _csr->csrE[nidx];
            // if(nid == 74655 || nid == 74684){continue;}
            if(dist_arr[nid] == -1){
                qPushBack(Q, nid);
                dist_arr[nid] = dist_arr[curID] + 1;

                #ifdef AfterD1Folding
                CC_ans += _csr->ff[nid] + dist_arr[nid] * _csr->representNode[nid];
                #else
                CC_ans += dist_arr[nid];
                #endif
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

    // int checkNodeID = 14294 ;
    // int checkNode_CC_ans = checkCC_Ans(csr, checkNodeID);
    // printf("CCs[%d] = %d\n", checkNodeID, checkNode_CC_ans);
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

    // int checkNode_CC_ans = checkCC_Ans(csr, checkNodeID);
    // printf("CCs[%d] = %d\n", checkNodeID, checkNode_CC_ans);
    // printf("ff[%d] = %d, w[%d] = %d\n", checkNodeID, csr->ff[checkNodeID], checkNodeID, csr->representNode[checkNodeID]);

    time1 = seconds();
    AP_detection(csr);
    time2 = seconds();
    AP_detectionTime = time2 - time1;
    printf("[Execution Time] AP_detection       = %f\n", AP_detectionTime);

    

    time1 = seconds();
    AP_Copy_And_Split(csr);
    // printf("compID[38621] = %d\n", csr->compID[38621]);
    time2 = seconds();
    AP_Copy_And_Split_Time = time2 - time1;
    printf("[Execution Time] AP_Copy_And_Split  = %f\n", AP_Copy_And_Split_Time);
    printf("apCount     = %8d\n", csr->ap_count);
    printf("compCount   = %8d\n", csr->compNum);
    printf("maxCompSize = %8d\n", csr->maxCompSize_afterSplit);

    
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