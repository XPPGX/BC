#include "AP_Process.h"

// #define AP_DEBUG

// #define AP_detection_DEBUG
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

        #ifdef AP_detection_DEBUG
        printf("_csr->ap_count = %d\n", ap_count);   
        #endif
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

// #define assignComponentID_DEBUG
// #define GetPartInfo_DEBUG
// #define sortAP_By_apNum_DEBUG
// #define Split_DEBUG
void AP_Copy_And_Split(struct CSR* _csr){
    _csr->compID = (int*)malloc(sizeof(int) * _csr->csrVSize);
    memset(_csr->compID, -1, sizeof(int) * _csr->csrVSize);
    _csr->maxCompSize_afterSplit = 0;
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
     * @todo 
     * 可能只要assign ap 的 neighbors 的 compID 就好
    */
    #pragma region assignComponentID

    int temp_compID = 0;
    for(int i = 0 ; i < ap_count ; i ++){
        apNodeID = _csr->AP_List[i];
        
        
        #ifdef assignComponentID_DEBUG
        printf("AP %d : \n", apNodeID);
        #endif

        for(apNeighborIndex = _csr->csrV[apNodeID] ; apNeighborIndex < _csr->oriCsrV[apNodeID + 1] ; apNeighborIndex ++){
            apNeighborNodeID = _csr->csrE[apNeighborIndex];

            /**
             * 如果某個 非AP 的 apNeighborNodeID 還沒有被走過，
             * 以 apNeighborNodeID 為起點 進行BFS，且遇到 AP 則不把 AP 塞進 Queue
            */
            if((_csr->compID[apNeighborNodeID] == -1) && (!(_csr->nodesType[apNeighborNodeID] & OriginAP))){

                int tempCompSize = 0;

                #ifdef assignComponentID_DEBUG
                printf("\t[new comp %d]\n", temp_compID);
                #endif

                qPushBack(Q, apNeighborNodeID);
                _csr->compID[apNeighborNodeID] = temp_compID;

                #ifdef assignComponentID_DEBUG
                printf("\t\t\t[comp %d] add node %d\n", temp_compID, apNeighborNodeID);
                #endif
                
                int currentNodeID   = -1;
                int neighborIndex   = -1;
                int neighborNodeID  = -1;
                while(!qIsEmpty(Q)){ 
                    currentNodeID = qPopFront(Q);

                    tempCompSize ++;
                    
                    #ifdef assignComponentID_DEBUG
                    printf("\t\tcurrentNodeID = %d\n", currentNodeID);
                    #endif

                    for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                        neighborNodeID = _csr->csrE[neighborIndex];

                        if((_csr->compID[neighborNodeID] == -1) && (!(_csr->nodesType[neighborNodeID] & OriginAP))){

                            #ifdef assignComponentID_DEBUG
                            printf("\t\t\t[comp %d] add node %d\n", temp_compID, neighborNodeID);
                            #endif

                            qPushBack(Q, neighborNodeID);
                            _csr->compID[neighborNodeID] = temp_compID;

                        }
                    }

                    
                }

                temp_compID ++;

                if(_csr->maxCompSize_afterSplit < tempCompSize){
                    _csr->maxCompSize_afterSplit = tempCompSize;
                }
            }
        }
    }

    int maxCompID = temp_compID;
    _csr->compNum = maxCompID;
    // for(int nodeID = _csr->startNodeID ; nodeID < _csr->endNodeID ; nodeID ++){
    //     if(mappingAP_ID[nodeID] != -1){
    //         printf("AP %d, index = %d\n", nodeID, mappingAP_ID[nodeID]);
    //     }
    // }

    #pragma endregion assignComponentID
    // printf("1\n");
    /**
     * We've got all compID of each nodes except for AP nodes so far
    */

    #pragma region sortAP_By_apNum
    //record that there are how many components are around a single AP u
    int compNum;

    //record the number of AP neighbors that AP u connected
    int apNeighborNum;

    //record the AP u connects to which components, if u connects to comp 0, then compFlag[0] is 1; else is 0. 
    int* compFlag           = (int*)malloc(sizeof(int) * maxCompID);

    //prepare two arrays "compNum_arr" and "apNeighborNum_arr"
    int* compNum_arr        = (int*)calloc(sizeof(int), _csr->csrVSize);
    int* apNeighborNum_arr  = (int*)calloc(sizeof(int), _csr->csrVSize);

    //紀錄每個 AP 最多連著幾個要分割的區域(AP點 或 component 都算)
    int maxBranch = 0;

    int cid = -1;
    for(int i = 0 ; i < ap_count ; i ++){
        apNodeID        = _csr->AP_List[i];

        compNum         = 0;
        apNeighborNum   = 0;
        memset(compFlag, 0, sizeof(int) * maxCompID);

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

        if(maxBranch < (compNum + apNeighborNum)){
            maxBranch = compNum + apNeighborNum;
        }
    }

    quicksort_nodeID_with_apNum(_csr->AP_List, apNeighborNum_arr, 0, ap_count - 1);

    #ifdef sortAP_By_apNum_DEBUG
    printf("maxBranch = %d\n", maxBranch);
    printf("AP(compN, apN) : \n");
    for(int i = 0 ; i < ap_count ; i ++){
        printf("\t%2d(%d, %d)\n", _csr->AP_List[i], compNum_arr[_csr->AP_List[i]], apNeighborNum_arr[_csr->AP_List[i]]);
    }
    printf("\n");
    #endif

    #pragma endregion //sortAP_By_apNum
    // printf("2\n");

    #pragma region AP_splitGraph
    /**
     * prepare for recording apClone information
    */
    int nextCsrE_offset = _csr->csrV[_csr->endNodeID + 1]; //從這個位置開始可以放新 node
    printf("nextCsrE_offset = %d\n", nextCsrE_offset);
    _csr->apCloneTrackOriAp_ID = (int*)malloc(sizeof(int) * (_csr->csrVSize) * 2);
    memset(_csr->apCloneTrackOriAp_ID, -1, sizeof(int) * (_csr->csrVSize) * 2);
    _csr->apCloneCount = 0;
    
    /**
     * prepare for record each part information of every originAP
    */
    struct part_info* part = (struct part_info*)malloc(sizeof(struct part_info) * maxBranch);

    int* dist_arr = (int*)malloc(sizeof(int) * (_csr->csrVSize) * 2);

    for(int i = ap_count - 1 ; i >= 0 ; i --){
        apNodeID = _csr->AP_List[i];

        /**
         * 如果 ap 周圍的 component part 被切開了，則對應的compNum數量就 --
         * 如果 ap 周圍的 apNeighbor 被切開了，則對應的 apNeighbor數量就 --
        */
        if(compNum_arr[apNodeID] == 0 && apNeighborNum_arr[apNodeID] == 0){
            continue;
        }

        #pragma region GetPartInfo
        
        #ifdef GetPartInfo_DEBUG
        printf("AP %d :\n", apNodeID);
        #endif

        memset(part, -1, sizeof(struct part_info) * maxBranch);
        memset(compFlag, -1, sizeof(int) * maxCompID);
        int total_represent = 0;
        int total_ff        = 0;
        
        int partIndex = 0;
        for(apNeighborIndex = _csr->csrV[apNodeID] ; apNeighborIndex < _csr->oriCsrV[apNodeID + 1] ; apNeighborIndex ++){
            apNeighborNodeID    = _csr->csrE[apNeighborIndex];
            int tempCompID      = _csr->compID[apNeighborNodeID]; //tempCompID = -1, if it is ap; else, if not
            
            #ifdef GetPartInfo_DEBUG
            printf("\tapNeighborNodeID = %d, tempCompID = %d\n", apNeighborNodeID, tempCompID);
            #endif

            /**
             * according to apNeighborNodeID is ap or comp to 
             * 1. assign apID or compID in part[partIndex]
             * 2. assign part[].represent
             * 3. assign part[].ff
            */
            if(tempCompID != -1 && compFlag[tempCompID] == -1){ //當 apNeighborNodeID u 不是AP 且 u的comp未被探訪
                compFlag[tempCompID]    = 1;
                part[partIndex].compID  = tempCompID;
                part[partIndex].apID    = -1;
                // partIndex ++;

                #ifdef GetPartInfo_DEBUG
                printf("\t\t[compID] %d\n", tempCompID);
                #endif

                //reset Q and dist_arr
                Q->front = 0;
                Q->rear = -1;
                memset(dist_arr, -1, sizeof(int) * (_csr->csrVSize) * 2);

                //Init source information
                dist_arr[apNodeID] = 0;
                int currentNodeID   = -1;
                int neighborIndex   = -1;
                int neighborNodeID  = -1;
                
                int part_represent = 0;
                int part_ff        = 0;

                //先對 apNodeID 周圍 compID == tempCompID 的 node 更新距離 與 part_info
                for(int neighborIndex = _csr->csrV[apNodeID] ; neighborIndex < _csr->oriCsrV[apNodeID + 1] ; neighborIndex ++){
                    int neighborNodeID = _csr->csrE[neighborIndex];
                    if(_csr->compID[neighborNodeID] == tempCompID){

                        #ifdef GetPartInfo_DEBUG
                        printf("\t\tneighborNodeID = %d, dist = 1\n", neighborNodeID);
                        #endif

                        qPushBack(Q, neighborNodeID);
                        dist_arr[neighborNodeID] = 1;
                        part_represent  += _csr->representNode[neighborNodeID];
                        part_ff         += _csr->ff[neighborNodeID] + dist_arr[neighborNodeID] * _csr->representNode[neighborNodeID];
                    }
                }
                // qPushBack(Q, apNodeID);
                
                
                while(!qIsEmpty(Q)){
                    currentNodeID = qPopFront(Q);
                    // printf("currentNodeID = %d : ", currentNodeID);
                    // printf("\tdist_arr[%d] = %d\n", currentNodeID, dist_arr[currentNodeID]);
                    for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                        neighborNodeID = _csr->csrE[neighborIndex];
                        /**
                         * 不用特別判斷 neighborNodeID 是否是 source(apNeighborNodeID)，
                         * 因為 dist_arr[apNeighborNodeID] == 0，apNeighborNodeID 是 起點
                        */
                        if(dist_arr[neighborNodeID] == -1){
                            // printf("%d, ", neighborNodeID);
                            qPushBack(Q, neighborNodeID);
                            dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;
                            part_represent += _csr->representNode[neighborNodeID];
                            part_ff        += _csr->ff[neighborNodeID] + dist_arr[neighborNodeID] * _csr->representNode[neighborNodeID];
                        }
                    }
                    // printf("\n");
                }
                
                #ifdef GetPartInfo_DEBUG
                printf("\t\t\tpart_represent = %d, part_ff = %d\n", part_represent, part_ff);
                #endif

                part[partIndex].represent   = part_represent;
                part[partIndex].ff          = part_ff;

                #ifdef GetPartInfo_DEBUG
                printf("\t\t\tpart[%d] = {represent = %d, ff = %d}\n", partIndex, part[partIndex].represent, part[partIndex].ff);
                #endif

                total_represent += part[partIndex].represent;
                total_ff += part[partIndex].ff;       

                partIndex ++;

                         
            }
            else if(tempCompID == -1){ //當 apNeighborNodeID u 是 AP 的時候
                part[partIndex].compID  = -1;
                part[partIndex].apID    = apNeighborNodeID;
                // partIndex ++;

                #ifdef GetPartInfo_DEBUG
                printf("\t\t[apNeighborNodeID] %d\n", apNeighborNodeID);
                #endif

                //reset Q and dist_arr
                Q->front = 0;
                Q->rear = -1;
                memset(dist_arr, -1, sizeof(int) * (_csr->csrVSize) * 2);

                //Init source information
                dist_arr[apNeighborNodeID]  = 0;
                qPushBack(Q, apNeighborNodeID);

                int currentNodeID   = -1;
                int neighborIndex   = -1;
                int neighborNodeID  = -1;
                
                int part_represent = 0;
                int part_ff        = 0;
                while(!qIsEmpty(Q)){
                    currentNodeID = qPopFront(Q);
                    // printf("\tdist_arr[%d] = %d\n", currentNodeID, dist_arr[currentNodeID]);
                    for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                        neighborNodeID = _csr->csrE[neighborIndex];

                        /**
                         * 不用特別判斷 neighborNodeID 是否是 source(apNeighborNodeID)，
                         * 因為 dist_arr[apNeighborNodeID] == 0，apNeighborNodeID 是 起點
                        */
                        if(dist_arr[neighborNodeID] == -1 && neighborNodeID != apNodeID){
                            qPushBack(Q, neighborNodeID);
                            dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;
                            part_represent += _csr->representNode[neighborNodeID];
                            part_ff        += _csr->ff[neighborNodeID] + dist_arr[neighborNodeID] * _csr->representNode[neighborNodeID];
                        }
                    }
                }
                
                #ifdef GetPartInfo_DEBUG
                printf("\t\t\tpart_represent = %d, part_ff = %d\n", part_represent, part_ff);
                #endif

                part[partIndex].represent   = part_represent + _csr->representNode[apNeighborNodeID];
                part[partIndex].ff          = part_ff + _csr->ff[apNeighborNodeID] + part[partIndex].represent;

                #ifdef GetPartInfo_DEBUG
                printf("\t\t\tpart[%d] = {represent = %d, ff = %d}\n", partIndex, part[partIndex].represent, part[partIndex].ff);
                #endif

                total_represent += part[partIndex].represent;
                total_ff += part[partIndex].ff; 

                partIndex ++;

            }
            else{ //碰到的node，其compID的資料已被創建
                #ifdef GetPartInfo_DEBUG
                printf("\t\t[compID] %d has been created\n", tempCompID);
                #endif
                continue;
            }

        }
        #pragma endregion //GetPartInfo

        #pragma region Split
        /**
         * apNodeID 已知 它對整個graph的dist，所以 CCs[apNodeID] 已經可得，
         * 所以之後 在各 component 內的 traverse 不會在計算 apNodeID本尊 的 CCs
        */
        _csr->CCs[apNodeID] = total_ff + _csr->ff[apNodeID];

        #ifdef Split_DEBUG
        printf("_csr->CCs[%d] = %d, total_ff = %d, total_represent = %d\n", apNodeID, _csr->CCs[apNodeID], total_ff, total_represent);
        #endif

        int partNum = compNum_arr[apNodeID] + apNeighborNum_arr[apNodeID];

        //暫存apNodeID_ori_ff, apNodeID_ori_represent
        int apNodeID_ori_ff         = _csr->ff[apNodeID];
        int apNodeID_ori_represent  = _csr->representNode[apNodeID];

        for(partIndex = 0 ; partIndex < partNum ; partIndex ++){
            //apNodeID主動，其他為被動
            

            if(part[partIndex].apID != -1){//Split target part is AP
                int apID = part[partIndex].apID;
                int outer_represent = total_represent - part[partIndex].represent + apNodeID_ori_represent;

                #ifdef Split_DEBUG
                printf("\tapID = %d, outer_represent = %d\n", apID, outer_represent);
                #endif

                //更新 represent 跟 ff
                _csr->representNode[apID]   += outer_represent;
                _csr->ff[apID]              += (total_ff - part[partIndex].ff) + outer_represent + apNodeID_ori_ff;

                #ifdef Split_DEBUG
                printf("\tapID %d = {w = %d, ff = %d}\n", apID, _csr->representNode[apID], _csr->ff[apID]);
                #endif

                //斷開 apNodeID 連到 apID 的 edge
                for(int nidx = _csr->csrV[apNodeID] ; nidx < _csr->oriCsrV[apNodeID + 1] ; nidx ++){
                    int nid = _csr->csrE[nidx];
                    if(nid == apID){
                        swap(&(_csr->csrE[nidx]), &(_csr->csrE[_csr->csrV[apNodeID]]));
                        _csr->csrV[apNodeID] ++;
                        _csr->csrNodesDegree[apNodeID] --;

                        #ifdef Split_DEBUG
                        printf("\tCut (%d, %d)\n", apNodeID, apID);
                        #endif

                        break;
                    }
                }

                //斷開 apID 連到 apNodeID 的 edge
                for(int nidx = _csr->csrV[apID] ; nidx < _csr->oriCsrV[apID + 1] ; nidx ++){
                    int nid = _csr->csrE[nidx];
                    if(nid == apNodeID){
                        swap(&(_csr->csrE[nidx]), &(_csr->csrE[_csr->csrV[apID]]));
                        _csr->csrV[apID] ++;
                        _csr->csrNodesDegree[apID] --;

                        #ifdef Split_DEBUG
                        printf("\tCut (%d, %d)\n", apID, apNodeID);
                        #endif

                        break;
                    }
                }
                //被動AP 的 apNeighborNum - 1
                apNeighborNum_arr[apID] --;
            }
            else if(part[partIndex].compID != -1){//Split target part is Comp
                //取得這個 component 以外的所有 node 個數，不包含 apNodeID
                int outer_represent = total_represent - part[partIndex].represent;

                #ifdef Split_DEBUG
                printf("\tcompID %d, outer_represent = %d\n", part[partIndex].compID, outer_represent);
                #endif

                /**
                 * 如果所有part之中，只有一個component，則
                 * 1. 不須創建 AP分身，用 AP本尊就可以
                 * 2. AP本尊.represent, AP本尊.ff，需要根據 outer_represent 去更新
                 * 3. 斷開 AP本尊 跟 其他 component 的 edges
                */
                if(compNum_arr[apNodeID] == 1){
                    _csr->representNode[apNodeID] += outer_represent;
                    _csr->ff[apNodeID] += (total_ff - part[partIndex].ff);

                    #ifdef Split_DEBUG
                    printf("\t[One comp] apNodeID %d = {represent = %d, ff = %d}\n", apNodeID, _csr->representNode[apNodeID], _csr->ff[apNodeID]);
                    #endif

                    /**
                     * apNodeID(AP本尊) 主動斷開 跟 (其他不同 comp 的 node) 的 edge
                     * (其他不同 comp 的 node) 主動斷開 跟 apNodeID(AP本尊) 的 edge
                    */
                    int tempNid = -1;
                    for(int nidx = _csr->csrV[apNodeID] ; nidx < _csr->oriCsrV[apNodeID + 1] ; nidx ++){
                        int nid = _csr->csrE[nidx];
                        tempNid = nid;
                        if(_csr->compID[nid] != part[partIndex].compID){
                            // apNodeID(AP本尊) 主動斷開 跟 (其他不同 comp 的 node(有可能也包含其他AP)) 的 edge
                            swap(&(_csr->csrE[nidx]), &(_csr->csrE[_csr->csrV[apNodeID]]));
                            _csr->csrV[apNodeID] ++;
                            _csr->csrNodesDegree[apNodeID] --;

                            #ifdef Split_DEBUG
                            printf("\tcut (%d, %d)\n", apNodeID, nid);
                            #endif

                            //(其他不同 comp 的 node(有可能也包含其他AP)) 主動斷開 跟 apNodeID(AP本尊) 的 edge
                            for(int nidx2 = _csr->csrV[nid] ; nidx2 < _csr->oriCsrV[nid + 1] ; nidx2 ++){
                                int nid2 = _csr->csrE[nidx2];
                                
                                if(nid2 == apNodeID){
                                    swap(&(_csr->csrE[nidx2]), &(_csr->csrE[_csr->csrV[nid]]));
                                    _csr->csrV[nid] ++;
                                    _csr->csrNodesDegree --;
                                    
                                    #ifdef Split_DEBUG
                                    printf("\tcut (%d, %d)\n", nid, nid2);
                                    #endif

                                    break;
                                }
                            }
                        }
                    }
                }
                
                /**
                 * @todo
                 * 如果所有part之中，有 k 個 component，則
                 * 1. 需要創建 k 個 AP分身，捨棄 AP本尊
                 * 2. 更新每個AP分身的 represent 跟 ff，根據 outer_represent 去更新
                 * 3. 以 _csr->endNodeID 開始去建立新的 node
                 * @bug
                 * 這裡的 _csr->csrV 處理有問題
                */
                else if(compNum_arr[apNodeID] > 1){
                    // int apCloneDataIndex = apClone->newNodeCount;
                    // apClone->newNodeCount ++;
                    _csr->apCloneCount ++;
                    int newApCloneID = _csr->endNodeID + _csr->apCloneCount;
                    _csr->nodesType[newApCloneID] = ClonedAP;
                    
                    #ifdef Split_DEBUG
                    printf("\tnewApCloneID = %d, ", newApCloneID);
                    #endif

                    // _csr->csrV[newApCloneID] = nextCsrV_offset;
                    
                    //新增 (apClone(ap分身) 的 ID) 跟 (ori_ap(ap本尊) 的 ID)
                    // apClone->Ori_apNodeID[apCloneDataIndex]     = apNodeID;
                    // apClone->apCloneID[apCloneDataIndex]        = _csr->endNodeID + apClone->newNodeCount;
                    _csr->apCloneTrackOriAp_ID[newApCloneID] = apNodeID;

                    //更新 apClone_ff, apClone_represent
                    // apClone->apCloneff[apCloneDataIndex]        = (total_ff - part[partIndex].ff) + _csr->ff[apNodeID];
                    // apClone->apCloneRepresent[apCloneDataIndex] = outer_represent + _csr->representNode[apNodeID];
                    _csr->representNode[newApCloneID]   = outer_represent + _csr->representNode[apNodeID];
                    _csr->ff[newApCloneID]              = (total_ff - part[partIndex].ff) + _csr->ff[apNodeID];

                    /**
                     * 此處的 
                     * 1. neighborNodeID 是 apNodeID 的其中一個neighbor
                     * 2. apNodeID 是 AP本尊
                     * 3. newAPcloneID 是 AP分身
                     * 
                     * 對 (compID[neighborNodeID] == part[partIndex].compID) 的 neighborNodeID
                     * "修改" neighborNodeID(主動) 與 apNodeID(被動) 的 edge => 變成 (neighborNodeID, newApCloneID)
                     * "新增" newAPcloneID 對 neighborNodeID 的 edge (newApCloneID, neighborNodeID)
                     * "移除" apNodeID 對 neighborNodeID 的 edge (apNodeID, neighborNodeID)
                    */
                    //讓 newApCloneID 指向 (apNodeID 當前指向的 csrE)
                    _csr->csrV[newApCloneID] = nextCsrE_offset;
                    _csr->oriCsrV[newApCloneID] = nextCsrE_offset;

                    #ifdef Split_DEBUG
                    printf("_csr->csrV[%d] = %d\n", newApCloneID, nextCsrE_offset);
                    #endif

                    for(int nidx = _csr->csrV[apNodeID] ; nidx < _csr->oriCsrV[apNodeID + 1] ; nidx ++){
                        int nid = _csr->csrE[nidx];
                        
                        //找 compID == part[partIndex].compID 的 neighbor
                        if(_csr->compID[nid] == part[partIndex].compID){
                            //[新增] (apCloneID, neighborNodeID)
                            _csr->csrE[nextCsrE_offset] = nid;

                            #ifdef Split_DEBUG
                            printf("\t\t[add] _csr->csrE[%d] = %d, ", nextCsrE_offset, _csr->csrE[nextCsrE_offset]);
                            #endif

                            nextCsrE_offset ++;

                            #ifdef Split_DEBUG
                            printf("\t\tnextCsrE_offset => %d\n", nextCsrE_offset);
                            #endif

                            //[移除] (apNodeID, neighborNodeID)
                            swap(&(_csr->csrE[nidx]), &(_csr->csrE[_csr->csrV[apNodeID]]));
                            _csr->csrV[apNodeID] ++;
                            _csr->csrNodesDegree[apNodeID] --;

                            #ifdef Split_DEBUG
                            printf("\t\t[cut] (%d, %d)\n", apNodeID, nid);
                            #endif

                            //[修改] (neighborNodeID, apNodeID) => (neighborNodeID, newApCloneID)
                            for(int nidx2 = _csr->csrV[nid] ; nidx2 < _csr->oriCsrV[nid + 1] ; nidx2 ++){
                                int nid2 = _csr->csrE[nidx2];
                                
                                if(nid2 == apNodeID){
                                    _csr->csrE[nidx2] = newApCloneID;

                                    #ifdef Split_DEBUG
                                    printf("\t\t[fix] (%d, %d) => (%d, %d)\n", nid, apNodeID, nid, newApCloneID);
                                    #endif

                                    break;
                                }
                            }
                        }
                    }
                    #ifdef Split_DEBUG
                    printf("\tnewApCloneID %d = {ff = %d, represent = %d, type = %x}\n\n", newApCloneID, _csr->ff[newApCloneID], _csr->representNode[newApCloneID], _csr->nodesType[newApCloneID]);
                    #endif
                }

            }
        }

        #ifdef Split_DEBUG
        printf("=============\n");
        #endif

        #pragma endregion //Split

        // #pragma region checkAns
        // /**
        //  * 檢查每個 ap node 是否都有抓到周圍的 part
        // */
        // // for(partIndex = 0 ; partIndex < (compNum_arr[apNodeID] + apNeighborNum_arr[apNodeID]) ; partIndex ++){
        // //     printf("\tpart[%d] = {compID = %d, apID = %d}\n", partIndex, part[partIndex].compID, part[partIndex].apID);
        // // }

        // /**
        //  * Check 分割的 ff, represent 是否正確
        // */
        // int curID   = -1;
        // int nIdx    = -1;
        // int nID     = -1;
        // int total_dist_from_apNode = 0;
        // Q->front = 0;
        // Q->rear = -1;
        // int checkNode = 7;
        // memset(dist_arr, -1, sizeof(int) * (_csr->csrVSize) * 2);

        // dist_arr[checkNode] = 0;
        // qPushBack(Q, checkNode);

        // while(!qIsEmpty(Q)){
        //     curID = qPopFront(Q);
        //     // printf("currentNodeID %d :\n", curID);
        //     for(nIdx = _csr->oriCsrV[curID] ; nIdx < _csr->oriCsrV[curID + 1] ; nIdx ++){
        //         nID = _csr->csrE[nIdx];

        //         if(dist_arr[nID] == -1){
        //             // printf("\tnID = %d\n", nID);
        //             qPushBack(Q, nID);
        //             dist_arr[nID] = dist_arr[curID] + 1;
        //             total_dist_from_apNode += dist_arr[nID];
        //         }
        //     }
        // }
        // printf("total_dist_from_apNode[%d] = %d\n", checkNode, total_dist_from_apNode);
        // #pragma endregion
        
        // printf("\n");
        // break;
    }

    
    /**
     * 最後要在 csrV, oriCsr都加上 csr的結尾
    */
    int theLastNodeID                   = _csr->endNodeID + _csr->apCloneCount;
    int theLastCsrE_offset              = nextCsrE_offset;
    _csr->csrV[theLastNodeID + 1]       = theLastCsrE_offset;
    _csr->oriCsrV[theLastNodeID + 1]    = theLastCsrE_offset;
    printf("[End csrV] _csr->csrV[%d] = %d\n", theLastNodeID + 1, _csr->csrV[theLastNodeID + 1]);
    #pragma endregion //AP_splitGraph
    // printf("3\n");
}