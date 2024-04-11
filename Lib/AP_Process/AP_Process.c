#include "AP_Process.h"

// #define AP_DEBUG

void AP_detection(struct CSR* _csr){
    _csr->AP_List               = (int*)malloc(sizeof(int) * _csr->csrVSize);
    _csr->AP_component_number   = (int*)malloc(sizeof(int) * _csr->csrVSize);
    memset(_csr->AP_List, -1, sizeof(int) * _csr->csrVSize);
    memset(_csr->AP_component_number, 0, sizeof(int) * _csr->csrVSize);

    int* parent         = (int*)malloc(sizeof(int) * _csr->csrVSize);
    int* depth_level    = (int*)malloc(sizeof(int) * _csr->csrVSize);
    int* low            = (int*)malloc(sizeof(int) * _csr->csrVSize);
    int* stack          = (int*)malloc(sizeof(int) * _csr->csrVSize);

    memset(parent, -1, sizeof(int) * _csr->csrVSize);
    memset(depth_level, 0, sizeof(int) * _csr->csrVSize);
    memset(low, 0, sizeof(int) * _csr->csrVSize);
    memset(stack, 0, sizeof(int) * _csr->csrVSize);


    int ap_count = 0;
    
    for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
        
        if(_csr->nodesType[nodeID] == D1){
            continue;
        }
        
        if(depth_level[nodeID]){
            continue;
        }

        // printf("new component\n");

        int depth = 1;

        int stack_index     = 0;
        
        int rootID          = nodeID;

        stack[stack_index]  = nodeID;

        depth_level[nodeID] = depth;
        low[nodeID]         = depth;
        depth ++;

        int currentNodeID   = -1;
        int neighborNodeID  = -1;
        int neighborIndex   = -1;
        while(stack_index >= 0){
            currentNodeID = stack[stack_index];
            // printf("\tcurrentNodeID = %2d\n", currentNodeID);
            //if neighbors are all visited, neighbor_all_visited = 1, else = 0;
            int neighbor_all_visited = 1;

            for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                neighborNodeID = _csr->csrE[neighborIndex];
                // printf("\tneighborNddeID = %2d\n", neighborNodeID);

                if(depth_level[neighborNodeID] == 0){
                    stack[++ stack_index]       = neighborNodeID;
                    parent[neighborNodeID]      = currentNodeID;
                    depth_level[neighborNodeID] = depth;
                    low[neighborNodeID]         = depth;

                    // printf("\t\tstack[%d] = %d, p = %d, d = %d, l = %d\n", stack_index, neighborNodeID, parent[neighborNodeID], depth_level[neighborNodeID], low[neighborNodeID]);

                    depth ++;

                    neighbor_all_visited = 0;

                    break;
                }
            }

            if(neighbor_all_visited){
                // printf("\t\tneighbor_all_visited\n");
                stack_index --;
                int childNum = 0;

                for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborNodeID = _csr->csrE[neighborIndex];

                    if(currentNodeID == parent[neighborNodeID]){
                        low[currentNodeID] = (low[currentNodeID] < low[neighborNodeID]) ? low[currentNodeID] : low[neighborNodeID];
                        childNum ++;
                        // printf("\t\t[update low][2] low[%d] = %d\n", currentNodeID, low[currentNodeID]);
                        
                        //這一段在判斷 root 是否是 AP
                        if((parent[currentNodeID] == -1) && (childNum > 1)){
                            _csr->AP_List[ap_count] = currentNodeID;
                            _csr->nodesType[currentNodeID] |= OriginAP;
                            ap_count ++;
                            break;
                        }

                        if(low[neighborNodeID] >= depth_level[currentNodeID] && (currentNodeID != rootID) && (!(_csr->nodesType[currentNodeID] & OriginAP))){
                            // printf("\t\tneighbor %d(%d, %d)...", neighborNodeID, depth_level[neighborNodeID], low[neighborNodeID]);

                            _csr->AP_List[ap_count ++] = currentNodeID;
                            _csr->nodesType[currentNodeID] |= OriginAP;
                            
                            // printf("\t\t[found AP] %d is AP!!!!!!!!!!!!!!\n", currentNodeID);
                        }
                    }
                    else if(neighborNodeID != parent[currentNodeID] && depth_level[neighborNodeID] < depth_level[currentNodeID]){
                        low[currentNodeID] = (low[currentNodeID] < depth_level[neighborNodeID]) ? low[currentNodeID] : depth_level[neighborNodeID];
                        // printf("\t\t[update low][1] low[%d] = %d\n", currentNodeID, low[currentNodeID]);

                    }
                    
                }
            }
        }

        #ifdef AP_DEBUG
        printf("AP : ");
        int* flag = (int*)calloc(sizeof(int), _csr->csrVSize);
        for(int i = 0 ; i < ap_count ; i ++){
            printf("%d, ", _csr->AP_List[i]);
            if(flag[_csr->AP_List[i]] == 1){
                printf("repeat AP %d\n", _csr->AP_List[i]);
                break;
            }
            else{
                flag[_csr->AP_List[i]] = 1;
            }
        }
        printf("\n");
        #endif
        //把AP的個數記錄在 _csr->ap_count
        _csr->ap_count = ap_count;
        printf("_csr->ap_count = %d\n", ap_count);   

    }

    free(parent);
    free(depth_level);
    free(low);
    free(stack);
}

void quicksort_nodeID_with_apNum(int* _nodes, int* _apNum, int _left, int _right){
    if(_left > _right){
        return;
    }
    int smallerAgent = _left;
    int smallerAgentNode = -1;
    int equalAgent = _left;
    int equalAgentNode = -1;
    int largerAgent = _right;
    int largerAgentNode = -1;

    int pivotNode = _nodes[_right];
    // printf("pivot : degree[%d] = %d .... \n", pivotNode, _nodeDegrees[pivotNode]);
    int tempNode = 0;
    while(equalAgent <= largerAgent){
        #ifdef DEBUG
        // printf("\tsmallerAgent = %d, equalAgent = %d, largerAgent = %d\n", smallerAgent, equalAgent, largerAgent);
        #endif

        smallerAgentNode = _nodes[smallerAgent];
        equalAgentNode = _nodes[equalAgent];
        largerAgentNode = _nodes[largerAgent];
        
        #ifdef DEBUG
        // printf("\tDegree_s[%d] = %d, Degree_e[%d] = %d, Degree_l[%d] = %d\n", smallerAgentNode, _nodeDegrees[smallerAgentNode], equalAgentNode, _nodeDegrees[equalAgentNode], largerAgentNode, _nodeDegrees[largerAgentNode]);
        #endif

        if(_apNum[equalAgentNode] < _apNum[pivotNode]){ //equalAgentNode的degree < pivotNode的degree
            // swap smallerAgentNode and equalAgentNode
            tempNode = _nodes[smallerAgent];
            _nodes[smallerAgent] = _nodes[equalAgent];
            _nodes[equalAgent] = tempNode;

            smallerAgent ++;
            equalAgent ++;
        }
        else if(_apNum[equalAgentNode] > _apNum[pivotNode]){ //equalAgentNode的degree > pivotNode的degree
            // swap largerAgentNode and equalAgentNode
            tempNode = _nodes[largerAgent];
            _nodes[largerAgent] = _nodes[equalAgent];
            _nodes[equalAgent] = tempNode;

            largerAgent --;
        }
        else{ //equalAgentNode的degree == pivotNode的degree
            equalAgent ++;
        }

    }
    
    // exit(1);
    #ifdef DEBUG
        
    #endif

    // smallerAgent現在是pivot key的開頭
    // largerAgent現在是pivotKey的結尾
    quicksort_nodeID_with_apNum(_nodes, _apNum, _left, smallerAgent - 1);
    quicksort_nodeID_with_apNum(_nodes, _apNum, largerAgent + 1, _right);
}

void AP_Copy_And_Split(struct CSR* _csr){
    _csr->compID = (int*)malloc(sizeof(int) * _csr->csrVSize);
    memset(_csr->compID, -1, sizeof(int) * _csr->csrVSize);

    int ap_count = _csr->ap_count;

    struct qQueue* Q = InitqQueue();
    qInitResize(Q, _csr->csrVSize);

    int apNeighborIndex   = -1;
    int apNeighborNodeID  = -1;
    int apNodeID        = -1;

    /**
     * 1. 先用 AP 走過整個 graph，並 assign 每個 node 一個 componentID
     * (AP node 不會 assign componentID)
     * 2. mapping AP to Index
    */
    
    #pragma region assignComponentID
    //mapping AP_ID
    int* mappingAP_ID = (int*)malloc(sizeof(int) * _csr->csrVSize);
    memset(mappingAP_ID, -1, sizeof(int) * _csr->csrVSize);
    
    int temp_compID = 0;
    for(int i = 0 ; i < ap_count ; i ++){
        apNodeID = _csr->AP_List[i];
        mappingAP_ID[apNodeID] = i;

        printf("AP %d : \n", apNodeID);
        for(apNeighborIndex = _csr->csrV[apNodeID] ; apNeighborIndex < _csr->oriCsrV[apNodeID + 1] ; apNeighborIndex ++){
            apNeighborNodeID = _csr->csrE[apNeighborIndex];

            /**
             * 如果某個 非AP 的 apNeighborNodeID 還沒有被走過，
             * 以 apNeighborNodeID 為起點 進行BFS，且遇到 AP 則不把 AP 塞進 Queue
            */
            if((_csr->compID[apNeighborNodeID] == -1) && (!(_csr->nodesType[apNeighborNodeID] & OriginAP))){
                printf("\t[new comp %d]\n", temp_compID);
                qPushBack(Q, apNeighborNodeID);
                _csr->compID[apNeighborNodeID] = temp_compID;

                printf("\t\t\t[comp %d] add node %d\n", temp_compID, apNeighborNodeID);

                int currentNodeID   = -1;
                int neighborIndex   = -1;
                int neighborNodeID  = -1;
                while(!qIsEmpty(Q)){ 
                    currentNodeID = qPopFront(Q);
                    printf("\t\tcurrentNodeID = %d\n", currentNodeID);
                    
                    for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                        neighborNodeID = _csr->csrE[neighborIndex];

                        if((_csr->compID[neighborNodeID] == -1) && (!(_csr->nodesType[neighborNodeID] & OriginAP))){
                            printf("\t\t\t[comp %d] add node %d\n", temp_compID, neighborNodeID);
                            qPushBack(Q, neighborNodeID);
                            _csr->compID[neighborNodeID] = temp_compID;

                        }
                    }

                    
                }

                temp_compID ++;
            }
        }
    }

    // for(int nodeID = _csr->startNodeID ; nodeID < _csr->endNodeID ; nodeID ++){
    //     if(mappingAP_ID[nodeID] != -1){
    //         printf("AP %d, index = %d\n", nodeID, mappingAP_ID[nodeID]);
    //     }
    // }

    #pragma endregion assignComponentID

    /**
     * We've got all compID of each nodes except for AP nodes so far
    */
    #pragma region sortAP_By_apNum
    //record that there are how many components are around a single AP u
    int compNum;

    //record the number of AP neighbors that AP u connected
    int apNeighborNum;

    //record the AP u connects to which components, if u connects to comp 0, then compFlag[0] is 1; else is 0. 
    int* compFlag           = (int*)malloc(sizeof(int) * temp_compID);

    //prepare two arrays "compNum_arr" and "apNeighborNum_arr"
    int* compNum_arr        = (int*)calloc(sizeof(int), _csr->csrVSize);
    int* apNeighborNum_arr  = (int*)calloc(sizeof(int), _csr->csrVSize);


    int cid = -1;
    for(int i = 0 ; i < ap_count ; i ++){
        apNodeID        = _csr->AP_List[i];

        compNum         = 0;
        apNeighborNum   = 0;
        memset(compFlag, 0, sizeof(int) * temp_compID);

        for(apNeighborIndex = _csr->csrV[apNodeID] ; apNeighborIndex < _csr->csrV[apNodeID + 1] ; apNeighborIndex ++){
            apNeighborNodeID    = _csr->csrE[apNeighborIndex];
            cid                 = _csr->compID[apNeighborNodeID];

            if((compFlag[cid] == 0)){
                
                if(cid != -1){
                    compFlag[cid] = 1;
                    compNum ++;
                }
                else{
                    apNeighborNum ++;
                }

            }
        }

        compNum_arr[apNodeID]          = compNum;
        apNeighborNum_arr[apNodeID]    = apNeighborNum;
    }

    quicksort_nodeID_with_apNum(_csr->AP_List, apNeighborNum_arr, 0, ap_count - 1);

    printf("AP(compN, apN) : \n");
    for(int i = 0 ; i < ap_count ; i ++){
        printf("\t%2d(%d, %d)\n", _csr->AP_List[i], compNum_arr[_csr->AP_List[i]], apNeighborNum_arr[_csr->AP_List[i]]);
    }
    printf("\n");
    #pragma endregion //sortAP_By_apNum
    

    #pragma region BridgeRemoval
    
    
    int* dist_arr = (int*)malloc(sizeof(int) * _csr->csrVSize);

    for(int i = ap_count - 1 ; i >= 0 ; i --){
        apNodeID = _csr->AP_List[i];
        printf("AP %d :\n", apNodeID);
        
        if(apNeighborNum_arr[apNodeID] > 0 && compNum_arr[apNodeID] == 0){

            for(apNeighborIndex = _csr->csrV[apNodeID] ; apNeighborIndex < _csr->oriCsrV[apNodeID] ; apNeighborIndex ++){
                apNeighborNodeID = _csr->csrE[apNeighborIndex];
                printf("\tap neighbor %d\n", apNeighborNodeID);
                
                //reset Q
                Q->front = 0;
                Q->rear = -1;
                
                //prepare dist_arr
                memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);

                //Init source info
                qPushBack(Q, apNeighborNodeID);
                dist_arr[apNeighborNodeID] = 0;

                int currentNodeID   = -1;
                int neighborNodeID  = -1;
                int neighborIndex   = -1;
                while(!qIsEmpty(Q)){
                    currentNodeID = qPopFront(Q);

                    for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                        neighborNodeID = _csr->csrE[neighborIndex];
                        
                    }
                }
            }
            
        }
        else{
            break;
        }
    }
    #pragma endregion //BridgeRemoval
}