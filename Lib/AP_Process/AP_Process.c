#include "AP_Process.h"

// #define AP_DEBUG
// #define assignComponentID_DEBUG

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
            }
        }
    }

    int maxCompID = temp_compID;
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

    printf("maxBranch = %d\n", maxBranch);
    printf("AP(compN, apN) : \n");
    for(int i = 0 ; i < ap_count ; i ++){
        printf("\t%2d(%d, %d)\n", _csr->AP_List[i], compNum_arr[_csr->AP_List[i]], apNeighborNum_arr[_csr->AP_List[i]]);
    }
    printf("\n");
    #pragma endregion //sortAP_By_apNum
    

    #pragma region AP_splitGraph
    struct part_info* part = (struct part_info*)malloc(sizeof(struct part_info) * maxBranch);

    int* dist_arr = (int*)malloc(sizeof(int) * _csr->csrVSize);

    for(int i = ap_count - 1 ; i >= 0 ; i --){
        apNodeID = _csr->AP_List[i];

        /**
         * 如果 ap 周圍的 component part 被切開了，則對應的compNum數量就 --
         * 如果 ap 周圍的 apNeighbor 被切開了，則對應的 apNeighbor數量就 --
        */
        if(compNum_arr[apNodeID] == 0 && apNeighborNum_arr[apNodeID] == 0){
            continue;
        }

        printf("AP %d :\n", apNodeID);
        memset(part, -1, sizeof(struct part_info) * maxBranch);
        memset(compFlag, -1, sizeof(int) * maxCompID);
        int total_represent = 0;
        int total_ff        = 0;

        int partIndex = 0;
        for(apNeighborIndex = _csr->csrV[apNodeID] ; apNeighborIndex < _csr->oriCsrV[apNodeID + 1] ; apNeighborIndex ++){
            apNeighborNodeID    = _csr->csrE[apNeighborIndex];
            int tempCompID      = _csr->compID[apNeighborNodeID]; //tempCompID = -1, if it is ap; else, if not
            
            printf("\tapNeighborNodeID = %d, tempCompID = %d\n", apNeighborNodeID, tempCompID);

            #pragma region GetPartInfo
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

                printf("\t\t[compID] %d\n", tempCompID);

                //reset Q and dist_arr
                Q->front = 0;
                Q->rear = -1;
                memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);

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
                        printf("\t\tneighborNodeID = %d, dist = 1\n", neighborNodeID);
                        qPushBack(Q, neighborNodeID);
                        dist_arr[neighborNodeID] = 1;
                        part_represent  += _csr->representNode[neighborNodeID];
                        part_ff         += _csr->ff[neighborNodeID] + dist_arr[neighborNodeID] * _csr->representNode[neighborNodeID];
                    }
                }
                // qPushBack(Q, apNodeID);
                
                
                while(!qIsEmpty(Q)){
                    currentNodeID = qPopFront(Q);
                    // printf("\tdist_arr[%d] = %d\n", currentNodeID, dist_arr[currentNodeID]);
                    for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                        neighborNodeID = _csr->csrE[neighborIndex];

                        /**
                         * 不用特別判斷 neighborNodeID 是否是 source(apNeighborNodeID)，
                         * 因為 dist_arr[apNeighborNodeID] == 0，apNeighborNodeID 是 起點
                        */
                        if(dist_arr[neighborNodeID] == -1){
                            qPushBack(Q, neighborNodeID);
                            dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;
                            part_represent += _csr->representNode[neighborNodeID];
                            part_ff        += _csr->ff[neighborNodeID] + dist_arr[neighborNodeID] * _csr->representNode[neighborNodeID];
                        }
                    }
                }
                printf("\t\t\tpart_represent = %d, part_ff = %d\n", part_represent, part_ff);

                part[partIndex].represent   = part_represent;
                part[partIndex].ff          = part_ff;

                printf("\t\t\tpart[%d] = {represent = %d, ff = %d}\n", partIndex, part[partIndex].represent, part[partIndex].ff);

                total_represent += part[partIndex].represent;
                total_ff += part[partIndex].ff;       

                partIndex ++;

                         
            }
            else if(tempCompID == -1){ //當 apNeighborNodeID u 是 AP 的時候
                part[partIndex].compID  = -1;
                part[partIndex].apID    = apNeighborNodeID;
                // partIndex ++;

                printf("\t\t[apNeighborNodeID] %d\n", apNeighborNodeID);

                //reset Q and dist_arr
                Q->front = 0;
                Q->rear = -1;
                memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);

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
                printf("\t\t\tpart_represent = %d, part_ff = %d\n", part_represent, part_ff);

                part[partIndex].represent   = part_represent + _csr->representNode[apNeighborNodeID];
                part[partIndex].ff          = part_ff + _csr->ff[apNeighborNodeID] + part[partIndex].represent;

                printf("\t\t\tpart[%d] = {represent = %d, ff = %d}\n", partIndex, part[partIndex].represent, part[partIndex].ff);

                total_represent += part[partIndex].represent;
                total_ff += part[partIndex].ff; 

                partIndex ++;

            }
            else{ //碰到的node，其compID的資料已被創建
                printf("\t\t[compID] %d has been created\n", tempCompID);
                continue;
            }

            #pragma endregion //GetPartInfo
            
        }

        #pragma region Split
        /**
         * apNodeID 已知 它對整個graph的dist，所以 CCs[apNodeID] 已經可得，
         * 所以之後 在各 component 內的 traverse 不會在計算 apNodeID本尊 的 CCs
        */
        _csr->CCs[apNodeID] = total_ff + _csr->ff[apNodeID];
        printf("_csr->CCs[%d] = %d, total_ff = %d, total_represent = %d\n", apNodeID, _csr->CCs[apNodeID], total_ff, total_represent);
        
        int partNum = compNum_arr[apNodeID] + apNeighborNum_arr[apNodeID];

        //暫存apNodeID_ori_ff, apNodeID_ori_represent
        int apNodeID_ori_ff         = _csr->ff[apNodeID];
        int apNodeID_ori_represent  = _csr->representNode[apNodeID];

        for(partIndex = 0 ; partIndex < partNum ; partIndex ++){
            //apNodeID主動，其他為被動
            

            if(part[partIndex].apID != -1){//Split AP
                int apID = part[partIndex].apID;
                int outer_represent = total_represent - part[partIndex].represent + apNodeID_ori_represent;
                printf("\tapID = %d, outer_represent = %d\n", apID, outer_represent);
                //更新 represent 跟 ff
                _csr->representNode[apID]   += outer_represent;
                _csr->ff[apID]              += (total_ff - part[partIndex].ff) + outer_represent + apNodeID_ori_ff;

                printf("\tapID %d = {w = %d, ff = %d}\n", apID, _csr->representNode[apID], _csr->ff[apID]);

                //斷開 apNodeID 連到 apID 的 edge
                for(int nidx = _csr->csrV[apNodeID] ; nidx < _csr->oriCsrV[apNodeID + 1] ; nidx ++){
                    int nid = _csr->csrE[nidx];
                    if(nid == apID){
                        swap(&(_csr->csrE[nidx]), &(_csr->csrE[_csr->csrV[apNodeID]]));
                        _csr->csrV[apNodeID] ++;
                        _csr->csrNodesDegree[apNodeID] --;

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

                        break;
                    }
                }
                //被動AP 的 apNeighborNum - 1
                apNeighborNum_arr[apID] --;
            }
            else if(part[partIndex].compID != -1){//Split Comp
                //取得這個 component 以外的所有 node 個數，不包含 apNodeID
                int outer_represent = total_represent - part[partIndex].represent;
                printf("\tcompID %d, outer_represent = %d\n", part[partIndex].compID, outer_represent);
                /**
                 * 如果所有part之中，只有一個component，則
                 * 1. 不須創建 AP分身，用 AP本尊就可以
                 * 2. AP本尊.represent, AP本尊.ff，需要根據 outer_represent 去更新
                 * 
                 * 如果所有part之中，有 k 個 component，則
                 * 1. 需要創建 k 個 AP分身，捨棄 AP本尊
                 * 2. 更新每個AP分身的 represent 跟 ff，根據 outer_represent 去更新
                */
                if(compNum_arr[apNodeID] == 1){
                    _csr->representNode[apNodeID] += outer_represent;
                    _csr->ff[apNodeID] += (total_ff - part[partIndex].ff);
                    printf("\t[One comp] apNodeID %d = {represent = %d, ff = %d}\n", apNodeID, _csr->representNode[apNodeID], _csr->ff[apNodeID]);
                }
                // else if(compNum_arr[apNodeID] > 1){
                    
                // }
            }
        }
        printf("=============\n");
        #pragma endregion //Split

        #pragma region checkAns
        /**
         * 檢查每個 ap node 是否都有抓到周圍的 part
        */
        // for(partIndex = 0 ; partIndex < (compNum_arr[apNodeID] + apNeighborNum_arr[apNodeID]) ; partIndex ++){
        //     printf("\tpart[%d] = {compID = %d, apID = %d}\n", partIndex, part[partIndex].compID, part[partIndex].apID);
        // }

        /**
         * Check 分割的 ff, represent 是否正確
        */
        int curID   = -1;
        int nIdx    = -1;
        int nID     = -1;
        int total_dist_from_apNode = 0;
        Q->front = 0;
        Q->rear = -1;
        int checkNode = 58;
        memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);

        dist_arr[checkNode] = 0;
        qPushBack(Q, checkNode);

        while(!qIsEmpty(Q)){
            curID = qPopFront(Q);

            for(nIdx = _csr->oriCsrV[curID] ; nIdx < _csr->oriCsrV[curID + 1] ; nIdx ++){
                nID = _csr->csrE[nIdx];

                if(dist_arr[nID] == -1){
                    qPushBack(Q, nID);
                    dist_arr[nID] = dist_arr[curID] + 1;
                    total_dist_from_apNode += dist_arr[nID];
                }
            }
        }
        printf("total_dist_from_apNode[%d] = %d\n", checkNode, total_dist_from_apNode);
        #pragma endregion
        printf("\n");
        // break;
    }

    // for(int i = _csr->startNodeID ; i <= _csr->endNodeID ; i ++){
    //     printf("compID[%d] = %d\n", i, _csr->compID[i]);
    // }
    #pragma endregion //AP_splitGraph
}