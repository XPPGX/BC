#include "AP_Process.h"

// #define AP_DEBUG

// #define AP_detection_DEBUG
void AP_detection(struct CSR* _csr){
    _csr->AP_List               = (int*)malloc(sizeof(int) * _csr->csrVSize);
    _csr->AP_component_number   = (int*)malloc(sizeof(int) * _csr->csrVSize);
    _csr->depth_level           = (int*)malloc(sizeof(int) * _csr->csrVSize);
    _csr->low                   = (int*)malloc(sizeof(int) * _csr->csrVSize);
    _csr->Dfs_sequence          = (int*)malloc(sizeof(int) * _csr->csrVSize);
    memset(_csr->AP_List, -1, sizeof(int) * _csr->csrVSize);
    memset(_csr->AP_component_number, 0, sizeof(int) * _csr->csrVSize);
    memset(_csr->depth_level, 0, sizeof(int) * _csr->csrVSize);
    memset(_csr->low, 0, sizeof(int) * _csr->csrVSize);
    memset(_csr->Dfs_sequence, -1, sizeof(int) * _csr->csrVSize);

    int* parent         = (int*)malloc(sizeof(int) * _csr->csrVSize);
    int* stack          = (int*)malloc(sizeof(int) * _csr->csrVSize);
    memset(parent, -1, sizeof(int) * _csr->csrVSize);
    memset(stack, 0, sizeof(int) * _csr->csrVSize);


    int ap_count = 0;
    
    for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
        
        if(_csr->nodesType[nodeID] & D1){
            continue;
        }
        
        if(_csr->depth_level[nodeID]){
            continue;
        }

        #ifdef AP_detection_DEBUG
        printf("new component\n");
        #endif

        int depth = 1;

        int stack_index     = 0;
        
        int rootID          = nodeID;

        stack[stack_index]  = nodeID;

        _csr->depth_level[nodeID]       = depth;
        _csr->low[nodeID]               = depth;
        _csr->Dfs_sequence[depth]   = nodeID;
        depth ++;

        int currentNodeID   = -1;
        int neighborNodeID  = -1;
        int neighborIndex   = -1;
        while(stack_index >= 0){
            currentNodeID = stack[stack_index];

            //if neighbors are all visited, neighbor_all_visited = 1, else = 0;
            int neighbor_all_visited = 1;

            for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                neighborNodeID = _csr->csrE[neighborIndex];
                // printf("\t\tneighborNddeID = %2d\n", neighborNodeID);

                if(_csr->depth_level[neighborNodeID] == 0){
                    stack[++ stack_index]               = neighborNodeID;
                    parent[neighborNodeID]              = currentNodeID;
                    _csr->depth_level[neighborNodeID]   = depth;
                    _csr->low[neighborNodeID]           = depth;
                    _csr->Dfs_sequence[depth]           = neighborNodeID;
                    #ifdef AP_detection_DEBUG
                    printf("\t\t\tstack[%d] = %d, p = %d, d = %d, l = %d\n", stack_index, neighborNodeID, parent[neighborNodeID], _csr->depth_level[neighborNodeID], _csr->low [neighborNodeID]);
                    #endif

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
                        #ifdef AP_detection_DEBUG
                        printf("\t\tchild neighbor %d\n", neighborNodeID);
                        #endif

                        _csr->low[currentNodeID] = (_csr->low[currentNodeID] < _csr->low[neighborNodeID]) ? _csr->low[currentNodeID] : _csr->low[neighborNodeID];
                        childNum ++;

                        #ifdef AP_detection_DEBUG
                        printf("\t\t[update _csr->low ][2] _csr->low [%d] = %d, neighborID = %d\n", currentNodeID, _csr->low [currentNodeID], neighborNodeID);
                        #endif

                        //這一段在判斷 root 是否是 AP
                        if((parent[currentNodeID] == -1) && (childNum > 1) && (!(_csr->nodesType[currentNodeID] & OriginAP))){
                            _csr->AP_List[ap_count] = currentNodeID;
                            _csr->nodesType[currentNodeID] |= OriginAP;
                            ap_count ++;
                            break;

                            #ifdef AP_detection_DEBUG
                            printf("\t\t[found AP] %d is AP!!!!!!!!!!!!!!\n", currentNodeID);
                            #endif
                        }

                        if(_csr->low [neighborNodeID] >= _csr->depth_level[currentNodeID] && (currentNodeID != rootID) && (!(_csr->nodesType[currentNodeID] & OriginAP))){
                            // printf("\t\tneighbor %d(%d, %d)...", neighborNodeID, _csr->depth_level[neighborNodeID], _csr->low [neighborNodeID]);

                            _csr->AP_List[ap_count ++] = currentNodeID;
                            _csr->nodesType[currentNodeID] |= OriginAP;
                            
                            #ifdef AP_detection_DEBUG
                            printf("\t\t[found AP] %d is AP!!!!!!!!!!!!!!\n", currentNodeID);
                            #endif
                        }
                    }
                    else if(neighborNodeID != parent[currentNodeID] && _csr->depth_level[neighborNodeID] < _csr->depth_level[currentNodeID]){
                        _csr->low[currentNodeID] = (_csr->low[currentNodeID] < _csr->depth_level[neighborNodeID]) ? _csr->low[currentNodeID] : _csr->depth_level[neighborNodeID];
                        
                        #ifdef AP_detection_DEBUG
                        printf("\t\t[update _csr->low ][1] _csr->low [%d] = %d, neighborID = %d\n", currentNodeID, _csr->low [currentNodeID], neighborNodeID);
                        #endif
                    }
                    
                }
                
            }
        }

        //把AP的個數記錄在 _csr->ap_count
        _csr->ap_count = ap_count;
        
        
        /**
         * 把同component，但low 值不同的 nodes 整合，並assign componentID 給每個 node
         * 
         * 整合後，low值相同的nodes，代表有可能在同一component，但實際是否在同一component
         * 還要用 BFS 配合 low值 去確認。
        */
        for(int depthIter = 1 ; depthIter < _csr->ordinaryNodeCount + 1 ; depthIter ++){
            int nodeID = _csr->Dfs_sequence[depthIter];

            #ifdef AP_detection_DEBUG
            printf("nodeID %d, ", nodeID);
            #endif

            if(_csr->nodesType[nodeID] & OriginAP){

                #ifdef AP_detection_DEBUG
                printf("is AP, childNum %d\n", childCompsNum[nodeID]);
                #endif

                continue;
            }
            
            
            int ancestorNodeID = _csr->Dfs_sequence[_csr->low[nodeID]];
            if(ancestorNodeID != nodeID && !(_csr->nodesType[ancestorNodeID] & OriginAP)){
                _csr->low[nodeID] = _csr->low[ancestorNodeID];

                #ifdef AP_detection_DEBUG
                printf("ancestorNodeID %d, low %d => update nodeID %d, low %d", ancestorNodeID, _csr->low[ancestorNodeID], nodeID, _csr->low[nodeID]);
                #endif

            }
            // else{

            //     #ifdef AP_detection_DEBUG
            //     printf("low %d", _csr->low[nodeID]);
            //     #endif
            // }

            #ifdef AP_detection_DEBUG
            printf("\n");
            #endif
        }

        #ifdef AP_detection_DEBUG
        printf("\n\n");
        #endif

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
        

        #ifdef AP_detection_DEBUG

        for(int depthIter = 1 ; depthIter < _csr->ordinaryNodeCount + 1 ; depthIter ++){
            printf("nodeID %d = {depth %d, low %d}", _csr->Dfs_sequence[depthIter], depthIter, _csr->low[_csr->Dfs_sequence[depthIter]]);
            if(_csr->nodesType[_csr->Dfs_sequence[depthIter]] & OriginAP){
                printf("\t AP");
            }
            printf("\n");
        }
        printf("_csr->ap_count = %d\n", ap_count);   
        #endif
    }

    free(parent);
    // free(_csr->depth_level);
    // free(_csr->low);
    free(stack);
}

void quicksort_nodeID_with_data(int* _nodes, int* _data, int _left, int _right){
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

        if(_data[equalAgentNode] < _data[pivotNode]){ //equalAgentNode的degree < pivotNode的degree
            // swap smallerAgentNode and equalAgentNode
            tempNode = _nodes[smallerAgent];
            _nodes[smallerAgent] = _nodes[equalAgent];
            _nodes[equalAgent] = tempNode;

            smallerAgent ++;
            equalAgent ++;
        }
        else if(_data[equalAgentNode] > _data[pivotNode]){ //equalAgentNode的degree > pivotNode的degree
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
    quicksort_nodeID_with_data(_nodes, _data, _left, smallerAgent - 1);
    quicksort_nodeID_with_data(_nodes, _data, largerAgent + 1, _right);
}

// #define assignPartId_for_AP_DEBUG
int assignPartId_for_AP(struct CSR* _csr, int* _partID, int _apNodeID, struct qQueue* _Q){
    // _partID 必須 已 reset 成 1
    // _Q 必須 已 reset

    int partCount = 0;
    
    #ifdef assignPartId_for_AP_DEBUG
    printf("ap = %d :\n", _apNodeID);
    #endif
    
    for(int nidx = _csr->csrV[_apNodeID] ; nidx < _csr->oriCsrV[_apNodeID + 1] ; nidx ++){
        int nid = _csr->csrE[nidx];
        
        if(_partID[nid] == -1){

            #ifdef assignPartId_for_AP_DEBUG
            printf("\t[new part %d]\n", partCount);
            #endif

            qPushBack(_Q, nid);
            _partID[nid] = partCount;

            #ifdef assignPartId_for_AP_DEBUG
            printf("\t\t\t[add to part %d] node %d\n", partCount, nid);
            #endif

            while(!qIsEmpty(_Q)){
                int curNodeID = qPopFront(_Q);

                #ifdef assignPartId_for_AP_DEBUG
                printf("\t\tcurID = %d\n", curNodeID);
                #endif

                for(int nidx2 = _csr->csrV[curNodeID] ; nidx2 < _csr->oriCsrV[curNodeID + 1] ; nidx2 ++){
                    int nid2 = _csr->csrE[nidx2];
                    
                    // printf("\t\t\tnid2 = %d\n", nid2);

                    if((_partID[nid2] == -1) && (nid2 != _apNodeID)){
                        qPushBack(_Q, nid2);
                        _partID[nid2] = partCount;

                        #ifdef assignPartId_for_AP_DEBUG
                        printf("\t\t\t[add to part %d] node %d\n", partCount, nid2);
                        #endif
                    }
                }
            }

            partCount ++;
        }
    }

    return partCount;
}

// #define findInterfaceAPs_DEBUG
void findInterfaceAPs(struct CSR* _csr, int* _partID, int* _eachPartNeighborNum, int* _partInterfaceAP, int _apNodeID){
    #ifdef findInterfaceAPs_DEBUG
    printf("AP %d : \n", _apNodeID);
    #endif
    
    for(int nidx = _csr->csrV[_apNodeID] ; nidx < _csr->oriCsrV[_apNodeID + 1] ; nidx ++){
        int nid         = _csr->csrE[nidx];
        int nidPartID   = _partID[nid];
        _eachPartNeighborNum[nidPartID] ++;
    }

    for(int nidx = _csr->csrV[_apNodeID] ; nidx < _csr->oriCsrV[_apNodeID + 1] ; nidx ++){
        int nid = _csr->csrE[nidx];
        
        if(_csr->compID[nid] == -1){ //如果 nid 是 AP
            int nidPartId = _partID[nid];
            if(_eachPartNeighborNum[nidPartId] == 1){
                _partInterfaceAP[nidPartId] = nid;

                #ifdef findInterfaceAPs_DEBUG
                printf("\t[Find Interface AP %d]\n", nid);
                #endif

            }
            #ifdef findInterfaceAPs_DEBUG
            if(_eachPartNeighborNum[nidPartId] > 1){
                // _ignoreOri_APs[nid] = 1;
                
                printf("\t[Ignore] ap neighbor %d\n", nid);
                
            }
            #endif
        }
    }
}

// #define getPartsInfo_DEBUG
int getPartsInfo(struct CSR* _csr, int* _partID, int _apNodeID, struct qQueue* _Q, struct part_info* _parts,
                     int _maxBranch, int* _partFlag, int* _dist_arr, int* _total_represent, int* _total_ff)
{
    int partIndex           = 0;
    _dist_arr[_apNodeID]    = 0;
    *_total_ff              = 0;
    *_total_represent       = 0;

    #ifdef getPartsInfo_DEBUG
    printf("AP %d : \n", _apNodeID);
    #endif

    for(int nidx = _csr->csrV[_apNodeID] ; nidx < _csr->oriCsrV[_apNodeID + 1] ; nidx ++){
        int nid = _csr->csrE[nidx];
        int nidPartID = _partID[nid];
        
        if(_partFlag[nidPartID] == 0){ //當 nidPartID 還沒有紀錄 partInfo

            _Q->front   = 0;
            _Q->rear    = -1;

            // printf("\t[part %d]\n", nidPartID);
            _partFlag[nidPartID] = 1;
            
            int part_represent  = 0;
            int part_ff         = 0;

            for(int nidx2 = _csr->csrV[_apNodeID] ; nidx2 < _csr->oriCsrV[_apNodeID + 1] ; nidx2 ++){
                int nid2 = _csr->csrE[nidx2];
                int nid2PartID = _partID[nid2];
                
                if(nid2PartID == nidPartID){
                    
                    _dist_arr[nid2] = 1;
                    qPushBack(_Q, nid2);
                    part_represent  += _csr->representNode[nid2];
                    part_ff         += _csr->ff[nid2] + _dist_arr[nid2] * _csr->representNode[nid2];

                    // printf("\t\tnid %d, dist = %d, w = %d, ff = %d, partID = %d, neighbor of AP\n", nid2, _dist_arr[nid2], _csr->representNode[nid2], _csr->ff[nid2], _partID[nid2]);
                }
            }

            while(!qIsEmpty(_Q)){
                int curID = qPopFront(_Q);

                for(int nidx3 = _csr->csrV[curID] ; nidx3 < _csr->oriCsrV[curID + 1] ; nidx3 ++){
                    int nid3 = _csr->csrE[nidx3];

                    if(_dist_arr[nid3] == -1){ //不會走回 _apNodeID
                        _dist_arr[nid3] = _dist_arr[curID] + 1;
                        qPushBack(_Q, nid3);
                        part_represent  += _csr->representNode[nid3];
                        part_ff         += _csr->ff[nid3] + _dist_arr[nid3] * _csr->representNode[nid3];
                        // printf("\t\tnid %d, dist = %d, w = %d, ff = %d, partID = %d\n", nid3, _dist_arr[nid3], _csr->representNode[nid3], _csr->ff[nid3], _partID[nid3]);
                    }
                }
            }

            _parts[partIndex].partID    = nidPartID;
            _parts[partIndex].represent = part_represent;
            _parts[partIndex].ff        = part_ff;

            #ifdef getPartsInfo_DEBUG
            printf("\tpart[%d] = {partID = %d, w = %d, ff = %d}\n", partIndex, _parts[partIndex].partID, _parts[partIndex].represent, _parts[partIndex].ff);
            #endif

            *_total_represent             += part_represent;
            *_total_ff                    += part_ff;

            partIndex ++;
        }
    }

    #ifdef getPartsInfo_DEBUG
    printf("\ttotal_w = %d, total_ff = %d\n", *_total_represent, *_total_ff);
    printf("\tinfo[%d] = {w = %d, ff = %d}\n", _apNodeID, _csr->representNode[_apNodeID], _csr->ff[_apNodeID]);
    #endif

    return partIndex; //回傳這個 AP 周圍共有幾個不同的 part
}

// #define assignComponentID_DEBUG
// #define sortAP_By_apNum_DEBUG
// #define GetPartInfo_DEBUG
// #define Split_DEBUG
void AP_Copy_And_Split(struct CSR* _csr){
    printf("==============================\n");
    printf("AP Copy and Split... ");
    _csr->compID = (int*)malloc(sizeof(int) * (_csr->csrVSize) * 2);
    memset(_csr->compID, -1, sizeof(int) * (_csr->csrVSize) * 2);
    _csr->maxCompSize_afterSplit = 0;
    int ap_count = _csr->ap_count;

    struct qQueue* Q = InitqQueue();
    qInitResize(Q, _csr->csrVSize);

    int apNeighborIndex   = -1;
    int apNeighborNodeID  = -1;
    int apNodeID        = -1;

    /**
     * 1. 先用 AP 走過整個 graph，並配合 low值 assign 每個 node 一個 componentID
     * (AP node 不會 assign componentID)
    */
    // double time1 = seconds();
    #pragma region assignComponentID
    int* visited = (int*)malloc(sizeof(int) * _csr->csrVSize);
    int compCounter = 0;
    for(int i = 0 ; i < ap_count ; i ++){
        apNodeID = _csr->AP_List[i];
        
        
        #ifdef assignComponentID_DEBUG
        printf("AP %d : \n", apNodeID);
        #endif

        for(apNeighborIndex = _csr->csrV[apNodeID] ; apNeighborIndex < _csr->oriCsrV[apNodeID + 1] ; apNeighborIndex ++){
            apNeighborNodeID = _csr->csrE[apNeighborIndex];

            /**
             * 如果某個 非AP 的 apNeighborNodeID 還沒有被走過，
             * 令 tempLow = _csr->low[apNeighborNodeID]，代表此次 traverse，所有(_csr->low[nodeID] == tempLow)的 nodeID 都有可能是同一個 component
             * 以 apNeighborNodeID 為起點 進行BFS，若過程中遇到 新的neighbor u，則把 u 塞入 Q 中
             * case 1. u 是 AP
             *      => 則不 assign compID 給 u
             * case 2. u 不是 AP
             *      => assign compID 給 u
             * 
             * 如果currentNodeID v，
             * case 3. v 是 AP
             *      => v 探訪自己的鄰居，並把 low[neighborNodeID] == tempLow 的 nodes加入 Q中，並 assign compID 給 neighborNodeID
             * case 4. v 不是 AP
             *      => v 探訪自己的鄰居，並做case 1 與 case 2的判斷與對應動作
            */
            if((_csr->compID[apNeighborNodeID] == -1) && (!(_csr->nodesType[apNeighborNodeID] & OriginAP))){
                memset(visited, 0, sizeof(int) * _csr->csrVSize);
                int tempLow = _csr->low[apNeighborNodeID];
                int tempCompSize = 0;

                #ifdef assignComponentID_DEBUG
                printf("\t[new comp %d]\n", compCounter);
                #endif

                //reset Q
                Q->front = 0;
                Q->rear = -1;

                qPushBack(Q, apNeighborNodeID);
                visited[apNeighborNodeID] = 1;
                _csr->compID[apNeighborNodeID] = compCounter;
                tempCompSize ++;
                
                #ifdef assignComponentID_DEBUG
                printf("\t\t\t[comp %d] add node %d\n", compCounter, apNeighborNodeID);
                #endif
                
                int currentNodeID   = -1;
                int neighborIndex   = -1;
                int neighborNodeID  = -1;
                while(!qIsEmpty(Q)){ 
                    currentNodeID = qPopFront(Q);
                    
                    #ifdef assignComponentID_DEBUG
                    printf("\t\tcurrentNodeID = %d\n", currentNodeID);
                    #endif

                    if(!(_csr->nodesType[currentNodeID] & OriginAP)){ //當 currentNodeID 不是 原生的AP
                        for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                            neighborNodeID = _csr->csrE[neighborIndex];
                            //&& (!(_csr->nodesType[neighborNodeID] & OriginAP))

                            if(_csr->compID[neighborNodeID] == -1 && neighborNodeID != apNodeID && visited[neighborNodeID] == 0){
                                
                                //可以推 AP 跟 普通node 進Q，但是不會assign compID 給 AP
                                qPushBack(Q, neighborNodeID);
                                visited[neighborNodeID] = 1;

                                if(!(_csr->nodesType[neighborNodeID] & OriginAP)){
                                    _csr->compID[neighborNodeID] = compCounter;
                                    tempCompSize ++;

                                    #ifdef assignComponentID_DEBUG
                                    printf("\t\t\t[comp %d] add node %d\n", compCounter, neighborNodeID);
                                    #endif
                                }
                            }
                        }
                    }
                    else{ //當 currentNodeID 是 原生的AP
                        for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                            neighborNodeID = _csr->csrE[neighborIndex];

                            if((_csr->compID[neighborNodeID] == -1) && (_csr->low[neighborNodeID] == tempLow) && (!(_csr->nodesType[neighborNodeID] & OriginAP)) && (neighborNodeID != apNodeID) && visited[neighborNodeID] == 0){
                                qPushBack(Q, neighborNodeID);
                                visited[neighborNodeID] = 1;
                                _csr->compID[neighborNodeID] = compCounter;
                                tempCompSize ++;

                                #ifdef assignComponentID_DEBUG
                                printf("\t\t\t[comp %d] add node %d\n", compCounter, neighborNodeID);
                                #endif
                            }
                        }
                    }
                }

                compCounter ++;

                if(_csr->maxCompSize_afterSplit < tempCompSize){
                    _csr->maxCompSize_afterSplit = tempCompSize;
                }
            }
        }
    }

    int compNumber = compCounter;
    _csr->compNum = compNumber;
    
    // printf("_csr->compNum = %d\n", _csr->compNum);
    // for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
    //     if(_csr->nodesType[nodeID] == D1){continue;}
    //     printf("\t_csr->compID[%d] = %d\n", nodeID, _csr->compID[nodeID]);
    // }
    free(visited);
    free(_csr->low);
    free(_csr->depth_level);
    free(_csr->Dfs_sequence);
    #pragma endregion assignComponentID
    // double time2 = seconds();
    // double assignComponentID_time = time2 - time1;
    // printf("[Execution Time] assignComponentID = %f\n", assignComponentID_time);
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
    int* compFlag           = (int*)malloc(sizeof(int) * compNumber);

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
        memset(compFlag, 0, sizeof(int) * compNumber);

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

    quicksort_nodeID_with_data(_csr->AP_List, apNeighborNum_arr, 0, ap_count - 1);

    #ifdef sortAP_By_apNum_DEBUG
    printf("maxBranch = %d\n", maxBranch);
    printf("AP(compN, apN) : \n");
    for(int i = 0 ; i < ap_count ; i ++){
        printf("\t%2d(%d, %d)\n", _csr->AP_List[i], compNum_arr[_csr->AP_List[i]], apNeighborNum_arr[_csr->AP_List[i]]);
    }
    printf("\n");
    #endif

    #pragma endregion sortAP_By_apNum
    // printf("2\n");
    

    #pragma region AP_splitGraph
    /**
     * prepare for recording apClone information
    */
    int nextCsrE_offset = _csr->csrV[_csr->endNodeID + 1]; //從這個位置開始可以放新 node
    // printf("nextCsrE_offset = %d\n", nextCsrE_offset);
    _csr->apCloneTrackOriAp_ID = (int*)malloc(sizeof(int) * (_csr->csrVSize) * 2);
    memset(_csr->apCloneTrackOriAp_ID, -1, sizeof(int) * (_csr->csrVSize) * 2);
    _csr->apCloneCount = 0;
    
    /**
     * prepare for record each part information of every originAP
    */
    struct part_info* parts = (struct part_info*)malloc(sizeof(struct part_info) * maxBranch);
    int* ignoreOri_APs = (int*)malloc(sizeof(int) * _csr->csrVSize); //如果某個oriAP u可以無視，則ignoreOri_APs[u] = 1
    int* partInterfaceAPs = (int*)malloc(sizeof(int) * maxBranch); //紀錄 partID 對應的 interfaceAP
    int* dist_arr = (int*)malloc(sizeof(int) * (_csr->csrVSize) * 2);
    int* partsID = (int*)malloc(sizeof(int) * (_csr->csrVSize) * 2); //紀錄每個nodes對 apNodeID的距離
    int* eachPartNeighborNum = (int*)malloc(sizeof(int) * maxBranch);
    int* partFlag = (int*)malloc(sizeof(int) * maxBranch);
    

    // printf("maxBranch = %d\n", maxBranch);

    for(int i = ap_count - 1 ; i >= 0 ; i --){
        apNodeID = _csr->AP_List[i];


        #pragma region GetPartInfo

        Q->front    = 0;
        Q->rear     = -1;
        memset(partsID, -1, sizeof(int) * (_csr->csrVSize) * 2);
        //取得 partID of each node for apNodeID
        int partNum = assignPartId_for_AP(_csr, partsID, apNodeID, Q);
        if(partNum == 1){
            #ifdef GetPartInfo_DEBUG
            printf("\n[Pass this AP] ap %d : only 1 part, w = %d, ff = %d   !!!!!!!!!!!!!!!!!!!\n\n", apNodeID, _csr->representNode[apNodeID], _csr->ff[apNodeID]);
            #endif

            continue;
        }



        memset(eachPartNeighborNum, 0, sizeof(int) * maxBranch);
        memset(partInterfaceAPs, -1, sizeof(int) * maxBranch);
        // memset(ignoreOri_APs, 0, sizeof(int) * _csr->csrVSize);
        //取得 ignoreOri_APs for apNodeID
        findInterfaceAPs(_csr, partsID, eachPartNeighborNum, partInterfaceAPs, apNodeID);



        Q->front = 0;
        Q->rear = -1;
        memset(partFlag, 0, sizeof(int) * maxBranch);
        memset(parts, -1, sizeof(struct part_info) * maxBranch);
        memset(dist_arr, -1, sizeof(int) * (_csr->csrVSize) * 2);
        //取得各 part 的 w 跟 ff，跟 total_w, total_ff, 還有 part 個數
        int total_represent = 0;
        int total_ff = 0;
        partNum = getPartsInfo(_csr, partsID, apNodeID, Q, parts, maxBranch, partFlag, dist_arr, &total_represent, &total_ff);

        #pragma endregion GetPartInfo


        #pragma region Split

        /**
         * 判斷是否要創建 AP分身，
         * 1. (如果eachPartNeighborNum中，有2個part的 neighbor個數 > 1)，代表 apNodeID需要創建 apClone => apCloneFlag = 1
         * 2. 否則，代表apNodeID不須創建 apClone => apCloneFlag = 0
        */
        int apCloneFlag = 0;
        int counter = 0;
        for(int partIndex = 0 ; partIndex < partNum ; partIndex ++){
            int partID = parts[partIndex].partID;

            #ifdef Split_DEBUG
            printf("\tpart %d neighborNum_of_ap = %d\n", partID, eachPartNeighborNum[partID]);   
            #endif

            if(eachPartNeighborNum[partID] > 1){
                counter ++;
            }

            if(counter > 1){
                apCloneFlag = 1;
                
                #ifdef Split_DEBUG
                printf("\t[Need to create AP Clone]\n");
                #endif
                
                break;
            }
        }

        
        /**
         * 開始切割 !!!!
         * 1. 如果只有一個 neighbor 是這個 partID，則該 neighbor 是 AP => handle AP
         * 2. 如果有多個 neighbor 是這個 partID，則該這些 neighbor(有可能有些是AP，有些不是) =>
         *      2.1 如果 apCloneFlag == 1，代表 "保留 AP本尊" 即可
         *      2.2 如果 apCloneFlag == 0，代表 "捨棄 AP本尊，創建 AP分身"
        */

        //先算好 apNodeID 的 CC，之後traverse的時候不會再算
        _csr->CCs[apNodeID] = total_ff + _csr->ff[apNodeID];

        #ifdef Split_DEBUG
        printf("CC[%d] = %d\n", apNodeID, _csr->CCs[apNodeID]);
        #endif

        int apNodeID_ori_represent  = _csr->representNode[apNodeID];
        int apNodeID_ori_ff         = _csr->ff[apNodeID];

        for(int partIndex = 0 ; partIndex < partNum ; partIndex ++){
            int partID = parts[partIndex].partID;
            
            if(eachPartNeighborNum[partID] == 1){ //這個 part 的接口只有一個點，則這個點是AP
                //取得對這個part而言，外部的資訊
                int outer_represent = total_represent - parts[partIndex].represent + apNodeID_ori_represent;
                int outer_ff        = (total_ff - parts[partIndex].ff) + (outer_represent + apNodeID_ori_ff);

                //取得要被斷開的 apID
                int apID = partInterfaceAPs[partID];

                //更新 represent, ff
                _csr->representNode[apID]   += outer_represent;
                _csr->ff[apID]              += outer_ff;
                
                #ifdef Split_DEBUG
                printf("\t[part Interface] ap %d = {w = %d, ff = %d}\n", apID, _csr->representNode[apID], _csr->ff[apID]);
                #endif

                //apNodeID 主動斷開 apID 的 edge，(apNodeID, apID)
                for(int nidx = _csr->csrV[apNodeID] ; nidx < _csr->oriCsrV[apNodeID + 1] ; nidx ++){
                    int nid = _csr->csrE[nidx];
                    if(nid == apID){
                        swap(&(_csr->csrE[nidx]), &(_csr->csrE[_csr->csrV[apNodeID]]));
                        _csr->csrV[apNodeID] ++;
                        _csr->csrNodesDegree[apNodeID] --;
                        
                        #ifdef Split_DEBUG
                        printf("\t\t[Cut] (%d, %d)\n", apNodeID, apID);
                        #endif

                        break;
                    }
                }

                //apID 主動斷開 apNodeID 的 edge，(apID, apNodeID)
                for(int nidx = _csr->csrV[apID] ; nidx < _csr->oriCsrV[apID + 1] ; nidx ++){
                    int nid = _csr->csrE[nidx];
                    if(nid == apNodeID){
                        swap(&(_csr->csrE[nidx]), &(_csr->csrE[_csr->csrV[apID]]));
                        _csr->csrV[apID] ++;
                        _csr->csrNodesDegree[apID] --;

                        #ifdef Split_DEBUG
                        printf("\t\t[Cut] (%d, %d)\n", apID, apNodeID);
                        #endif

                        break;
                    }
                }
            }
            else if(eachPartNeighborNum[partID] > 1){ //這個 part 的接口有多個點，要再判斷是否只有一個這種part
                if(apCloneFlag == 0){ //如果這種 part 只有 1個，則只需更新 AP本尊 的資訊就好
                    //取得對這個part而言，外部的資訊
                    int outer_represent = total_represent - parts[partIndex].represent;
                    int outer_ff = total_ff - parts[partIndex].ff;
                    
                    //更新 apNodeID 的 represent 跟 ff
                    _csr->representNode[apNodeID] += outer_represent;
                    _csr->ff[apNodeID] += outer_ff;

                    #ifdef Split_DEBUG
                    printf("\t[one part] apNodeID %d = {w = %d, ff = %d}\n", apNodeID, _csr->representNode[apNodeID], _csr->ff[apNodeID]);
                    #endif

                    /**
                     * apNodeID(ap本尊) 主動斷開 跟 (其他不同 partID 的 node) 的 edge
                     * (其他不同 comp 的 node) 主動斷開 跟 apNodeID(AP本尊) 的 edge
                    */
                    for(int nidx = _csr->csrV[apNodeID] ; nidx < _csr->oriCsrV[apNodeID + 1] ; nidx ++){
                        int nid = _csr->csrE[nidx];

                        if(partsID[nid] != partID){
                            //apNodeID(AP本尊) 主動斷開 跟 (其他不同 partID 的 node u) 的 edge (apNodeID, u)，u有可能是不同part的AP
                            swap(&(_csr->csrE[nidx]), &(_csr->csrE[_csr->csrV[apNodeID]]));
                            _csr->csrV[apNodeID] ++;
                            _csr->csrNodesDegree[apNodeID] --;

                            #ifdef Split_DEBUG
                            printf("\t\t[Cut] (%d, %d)\n", apNodeID, nid);
                            #endif

                            //其他不同 partID 的 node u 主動斷開 跟 apNodeID(AP本尊) 的 edge (u, apNodeID)
                            for(int nidx2 = _csr->csrV[nid] ; nidx2 < _csr->oriCsrV[nid + 1] ; nidx2 ++){
                                int nid2 = _csr->csrE[nidx2];

                                if(nid2 == apNodeID){
                                    swap(&(_csr->csrE[nidx2]), &(_csr->csrE[_csr->csrV[nid]]));
                                    _csr->csrV[nid] ++;
                                    _csr->csrNodesDegree[nid] --;

                                    #ifdef Split_DEBUG
                                    printf("\t\t[Cut] (%d, %d)\n", nid, nid2);
                                    #endif

                                    break;
                                }
                            }
                        }
                    }
                }
                else{ //如果這種 part 有多個，則需要 捨棄 AP本尊，並創建 AP分身
                    
                    //取得對這個part而言，外部的資訊
                    int outer_represent = total_represent - parts[partIndex].represent;
                    int outer_ff        = total_ff - parts[partIndex].ff;
                    
                    #ifdef Split_DEBUG
                    printf("\touter_represent = %d, outer_ff = %d\n", outer_represent, outer_ff);
                    #endif

                    //創建新的 apClone
                    _csr->apCloneCount ++;
                    int newApCloneID = _csr->endNodeID + _csr->apCloneCount;

                    _csr->nodesType[newApCloneID] |= ClonedAP;

                    _csr->apCloneTrackOriAp_ID[newApCloneID] = apNodeID;

                    
                    //assign newApCloneID 的 represent 跟 ff
                    _csr->representNode[newApCloneID]   = outer_represent + _csr->representNode[apNodeID];
                    _csr->ff[newApCloneID]              = outer_ff + _csr->ff[apNodeID];

                    
                    /**
                     * 此處的 
                     * 1. nid 是 apNodeID 的其中一個neighbor
                     * 2. apNodeID 是 AP本尊
                     * 3. newAPcloneID 是 AP分身
                     * 
                     * 對 (partsID[nid] == partID) 的 nid
                     * "修改" nid(主動) 與 apNodeID(被動) 的 edge => 變成 (nid, newApCloneID)
                     * "新增" newAPcloneID 對 nid 的 edge (newApCloneID, nid)
                     * "移除" apNodeID 對 nid 的 edge (apNodeID, nid)
                    */
                    
                    //讓 newApCloneID 指向 (apNodeID 當前指向的 csrE)
                    _csr->csrV[newApCloneID] = nextCsrE_offset;
                    _csr->oriCsrV[newApCloneID] = nextCsrE_offset;

                    #ifdef Split_DEBUG
                    printf("\t[New AP Clone] _csr->csrV[%d] = %d\n", newApCloneID, _csr->csrV[newApCloneID]);
                    #endif

                    for(int nidx = _csr->csrV[apNodeID] ; nidx < _csr->oriCsrV[apNodeID + 1] ; nidx ++){
                        int nid = _csr->csrE[nidx];

                        //找 partsID[nid] == partID 的 nid
                        if(partsID[nid] == partID){
                            //[新增] edge，(newApCloneID, nid)
                            _csr->csrE[nextCsrE_offset] = nid;

                            #ifdef Split_DEBUG
                            printf("\t\t[add] csrE[%d] = %d, ", nextCsrE_offset, _csr->csrE[nextCsrE_offset]);
                            #endif

                            nextCsrE_offset ++;
                            
                            #ifdef Split_DEBUG
                            printf("\t\tnextCsrE_offset => %d\n", nextCsrE_offset);
                            #endif

                            //[移除] edge (apNodeID, nid)
                            swap(&(_csr->csrE[nidx]), &(_csr->csrE[_csr->csrV[apNodeID]]));
                            _csr->csrV[apNodeID] ++;
                            _csr->csrNodesDegree[apNodeID] --;

                            #ifdef Split_DEBUG
                            printf("\t\t[cut] (%d, %d)\n", apNodeID, nid);
                            #endif

                            //[修改] edge (nid, apNodeID) => (nid, newApCloneID)
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
                    printf("\tnewApCloneID %d = {w = %d, ff = %d, type = %x, csrE_offset = %d}\n\n", newApCloneID, _csr->representNode[newApCloneID], _csr->ff[newApCloneID], _csr->nodesType[newApCloneID], _csr->csrV[newApCloneID]);
                    #endif

                    //紀錄 csrV的結尾
                    _csr->csrV[newApCloneID + 1] = nextCsrE_offset;
                    _csr->oriCsrV[newApCloneID + 1] = nextCsrE_offset;
                    
                    
                    // printf("newApCloneID = %d, csrV = %d, N = {", newApCloneID, _csr->csrV[newApCloneID]);
                    // for(int nidx = _csr->csrV[newApCloneID] ; nidx < _csr->oriCsrV[newApCloneID + 1] ; nidx ++){
                    //     int nid = _csr->csrE[nidx];
                    //     printf("%d, ", nid);
                    // }
                    // printf("}\n");
                }
            }
        }
        #pragma endregion Split
    }

    //更新 endNodeID
    _csr->endNodeID = _csr->endNodeID + _csr->apCloneCount;
    #pragma endregion //AP_splitGraph
    
    // printf("3\n");
    printf("Finished.\n");
    printf("apCount         = %8d\n", _csr->ap_count);
    printf("compCount       = %8d\n", _csr->compNum);
    printf("maxCompSize     = %8d\n", _csr->maxCompSize_afterSplit);
    printf("endNodeID       = %8d\n", _csr->endNodeID);
    printf("==============================\n");

    free(Q->dataArr);
    free(Q);

    free(compFlag);
    free(compNum_arr);
    free(apNeighborNum_arr);

    free(parts);
    free(ignoreOri_APs);
    free(partInterfaceAPs);
    free(dist_arr);
    free(partsID);
    free(eachPartNeighborNum);
    free(partFlag);
}