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

    int checkNodeID = 38;
    int checkNode_CC_ans = checkCC_Ans(csr, checkNodeID);
    printf("CCs[%d] = %d\n", checkNodeID, checkNode_CC_ans);
    

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
    printf("apCount     = %8d\n", csr->ap_count);
    printf("compCount   = %8d\n", csr->compNum);
    printf("maxCompSize = %8d\n", csr->maxCompSize_afterSplit);
    printf("endNodeID   = %8d\n", csr->endNodeID);
    
    struct newID_info* newID_infos = rebuildGraph(csr);

    int* dist_arr   = (int*)malloc(sizeof(int) * csr->csrVSize * 2);
    int* nodeQ      = (int*)malloc(sizeof(int) * csr->csrVSize * 2);
    int Q_front     = 0;
    int Q_rear      = -1;
    for(int sourceNewID = 0 ; sourceNewID <= csr->newEndID ; sourceNewID ++){
        int oldID = csr->mapNodeID_New_to_Old[sourceNewID];
        int sourceType = csr->nodesType[oldID];

        if(sourceType & ClonedAP){
            printf("newID %d, oldID %d, type %x\n", sourceNewID, oldID, sourceType);
            continue;
        }

        Q_front = 0;
        Q_rear = -1;
        memset(dist_arr, -1, sizeof(int) * csr->csrVSize * 2);
        
        dist_arr[sourceNewID] = 0;
        nodeQ[++Q_rear] = sourceNewID;
        int allDist = 0;
        while(!(Q_front > Q_rear)){
            int newCurID = nodeQ[Q_front++];
            
            for(int new_nidx = csr->orderedCsrV[newCurID] ; new_nidx < csr->orderedCsrV[newCurID + 1] ; new_nidx ++){
                int new_nid = csr->orderedCsrE[new_nidx];
                
                if(dist_arr[new_nid] == -1){
                    dist_arr[new_nid] = dist_arr[newCurID] + 1;
                    nodeQ[++Q_rear] = new_nid;

                    allDist += newID_infos[new_nid].ff + dist_arr[new_nid] * newID_infos[new_nid].w;
                }
            }
        }
        csr->CCs[oldID] = allDist + csr->ff[oldID];
        printf("CC[%d] = %d\n", oldID, csr->CCs[oldID]);
    }
}