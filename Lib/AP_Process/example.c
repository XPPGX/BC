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

void CC(struct CSR* _csr, int* _trueAns){
    int* nodeQ = (int*)malloc(sizeof(int) * _csr->csrVSize);
    int Q_front = 0;
    int Q_rear = -1;
    
    int* dist_arr = (int*)malloc(sizeof(int) * _csr->csrVSize);
    
    for(int sourceID = _csr->startNodeID ; sourceID <= _csr->endNodeID ; sourceID ++){
        Q_front = 0;
        Q_rear = -1;
        memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);

        dist_arr[sourceID] = 0;
        nodeQ[++Q_rear] = sourceID;

        register int curID      = -1;
        register int nID        = -1;
        register int nidx       = -1;
        register int allDist    = 0;
        while(!(Q_front > Q_rear)){
            curID = nodeQ[Q_front++];
            
            for(nidx = _csr->csrV[curID] ; nidx < _csr->csrV[curID + 1] ; nidx ++){
                nID = _csr->csrE[nidx];

                if(dist_arr[nID] == -1){
                    nodeQ[++Q_rear] = nID;
                    dist_arr[nID] = dist_arr[curID] + 1;

                    allDist += dist_arr[nID];
                }
            }
        }
        _trueAns[sourceID] = allDist;
        // printf("trueAns[%d] = %d\n", sourceID, _trueAns[sourceID]);
    }
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

    int* trueAns = (int*)malloc(sizeof(int) * csr->csrVSize);
    CC(csr, trueAns);
    // int checkNodeID = 55738;
    // int checkNode_CC_ans = checkCC_Ans(csr, checkNodeID);
    // printf("CCs[%d] = %d\n", checkNodeID, checkNode_CC_ans);
    

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
    
    struct newID_info* newID_infos = rebuildGraph(csr);

    int* dist_arr   = (int*)malloc(sizeof(int) * csr->csrVSize * 2);
    int* nodeQ      = (int*)malloc(sizeof(int) * csr->csrVSize * 2);
    int Q_front     = 0;
    int Q_rear      = -1;
    for(int sourceNewID = 0 ; sourceNewID <= csr->newEndID ; sourceNewID ++){
        int oldID = csr->mapNodeID_New_to_Old[sourceNewID];
        int sourceType = csr->nodesType[oldID];
        // printf("sourceOldID = %d\n", oldID);
        if(sourceType & ClonedAP){
            // printf("newID %d, oldID %d, type %x\n", sourceNewID, oldID, sourceType);
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
    }
    
    #pragma region d1GetCC_FromParent
    int d1NodeID        = -1;
    int d1NodeParentID  = -1;
    for(int d1NodeIndex = csr->degreeOneNodesQ->rear ; d1NodeIndex >= 0 ; d1NodeIndex --){
        d1NodeID        = csr->degreeOneNodesQ->dataArr[d1NodeIndex];
        d1NodeParentID  = csr->D1Parent[d1NodeID];
        csr->CCs[d1NodeID]  = csr->CCs[d1NodeParentID] + csr->totalNodeNumber - 2 * csr->representNode[d1NodeID];
    }
    printf("\n");
    #pragma endregion d1GetCC_FromParent


    for(int nodeID = csr->startNodeID ; nodeID <= (csr->endNodeID - csr->apCloneCount) ; nodeID ++){
        if(csr->CCs[nodeID] != trueAns[nodeID]){
            printf("[ERROR] CC[%d] = %d, trueAns[%d] = %d\n", nodeID, csr->CCs[nodeID], nodeID, trueAns[nodeID]);
            if(csr->nodesType[nodeID] & D1){
                printf("\tnodeID %d is D1\n", nodeID);
            }
            // exit(1);
        }
    }
    printf("[Success]\n");
}