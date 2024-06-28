#include "D1Process.h"

// #define DEBUG_D1

void D1Folding(struct CSR* _csr){
    printf("==============================\n");
    printf("D1 Folding...");

    _csr->hubNodeCount          = 0;
    _csr->ordinaryNodeCount     = 0;
    _csr->foldedDegreeOneCount  = 0;
    struct qQueue* d1Q      = _csr->degreeOneNodesQ;
    

    int d1NodeID            = -1;
    int d1ParentID          = -1;
    int d1ParentNeighborID  = -1;
    
    int outerNodesNum       = 0;

    _csr->representNode     = (int*)malloc(sizeof(int) * (_csr->csrVSize) * 2);
    for(int i = 0 ; i < _csr->csrVSize ; i ++){
        _csr->representNode[i] = 1;
    }
    _csr->ff                = (int*)calloc(sizeof(int), (_csr->csrVSize) * 2);

    //record each d1Node's ancestor to recover the dist of each d1Node quickly
    _csr->D1Parent          = (int*)malloc(sizeof(int) * (_csr->csrVSize)); //Note : its size is |V|
    memset(_csr->D1Parent, -1, sizeof(int) * _csr->foldedDegreeOneCount);

    #pragma region d1Folding
    while(!qIsEmpty(d1Q)){
        d1NodeID                    = qPopFront(d1Q);
        _csr->nodesType[d1NodeID]   = D1;

        d1ParentID                  = _csr->csrE[_csr->csrV[d1NodeID]];

        _csr->D1Parent[d1NodeID]    = d1ParentID;

        _csr->nodesType[d1ParentID] = D1Hub;
        
        //Update reach value of represent node, formula is "represent[parent] = represent[parent] + represent[d1Node]"
        _csr->representNode[d1ParentID] = _csr->representNode[d1ParentID] + _csr->representNode[d1NodeID];
        
        //Update ff value of represent node, formula is "ff[parent] = ff[parent] + represent[d1Node] * 1 + ff[d1Node]"
        _csr->ff[d1ParentID] = _csr->ff[d1ParentID] + _csr->representNode[d1NodeID] + _csr->ff[d1NodeID];

        #ifdef DEBUG_D1
        printf("%d, linking to %d\n", d1NodeID, d1ParentID);
        #endif
        
        for(int d1ParentNeighborIndex = _csr->csrV[d1ParentID] ; d1ParentNeighborIndex < _csr->oriCsrV[d1ParentID + 1] ; d1ParentNeighborIndex ++){
            d1ParentNeighborID = _csr->csrE[d1ParentNeighborIndex];
            if(d1ParentNeighborID == d1NodeID){
                swap(&(_csr->csrE[d1ParentNeighborIndex]), &(_csr->csrE[_csr->csrV[d1ParentID]]));
                break;
            }
        }

        // d1 node will not change its last csrV offset for find the last parent quickly !!
        
        // d1Parent的offset往後移動一格，代表刪除d1Node
        _csr->csrV[d1ParentID] ++;
        // d1Parent的degree - 1;
        _csr->csrNodesDegree[d1ParentID] --;
        // 檢查parent是否也變成d1, 是的話要推進Q裡面
        if(_csr->csrNodesDegree[d1ParentID] == 1){
            qPushBack(d1Q, d1ParentID);
        }
        
        //計數有多少D1 Nodes
        _csr->foldedDegreeOneCount ++;
    }
    #pragma endregion //Folding




    // #pragma region record_d1Node_parent
    // //record each d1Node's ancestor to recover the dist of each d1Node quickly
    // _csr->D1Parent = (int*)malloc(sizeof(int) * (_csr->csrVSize)); //Note : its size is |V|
    // memset(_csr->D1Parent, -1, sizeof(int) * _csr->foldedDegreeOneCount);
    // printf("\n");
    // for(int iter = _csr->degreeOneNodesQ->rear ; iter >= 0 ; iter --){
    //     int d1NodeID        = _csr->degreeOneNodesQ->dataArr[iter];
    //     int d1NodeParentID  = _csr->csrE[_csr->csrV[d1NodeID]];
        
    //     _csr->D1Parent[d1NodeID] = d1NodeParentID;

    //     #ifdef DEBUG_D1
    //     printf("D1nodeID = %2d, Parent = %2d\n", d1NodeID, d1NodeParentID);
    //     #endif

    // }
    // #pragma endregion //record_d1Node_parent

    #pragma region PrepareNodeList
    _csr->notD1Node = (int*)malloc(sizeof(int) * (_csr->endNodeID - _csr->startNodeID + 1 - _csr->foldedDegreeOneCount));
    int notD1NodeIndex = 0;
    //Counting and preparing the arrays of different type node'
    #ifdef DEBUG_D1
    printf("nodeID[nodeID] = (represent, ff, csrOffset)\n");
    #endif
    
    for(int i = _csr->startNodeID ; i <= _csr->endNodeID ; i ++){
        
        switch(_csr->nodesType[i]){

            case D1Hub:
                _csr->hubNodeCount ++;
                _csr->notD1Node[notD1NodeIndex] = i;

                #ifdef DEBUG_D1
                printf("notD1Node[%2d] = %2d\n", notD1NodeIndex, i);
                #endif

                notD1NodeIndex ++;
                _csr->ordinaryNodeCount ++;
                break;

            case D1:
                break;

            default:
                _csr->nodesType[i] = Ordinary;
                _csr->notD1Node[notD1NodeIndex] = i;

                #ifdef DEBUG_D1
                printf("notD1Node[%2d] = %2d\n", notD1NodeIndex, i);
                #endif

                notD1NodeIndex ++;
                _csr->ordinaryNodeCount ++;
                break;

        }
        
        #ifdef DEBUG_D1
        printf("nodeID[%2d] = (%2d, %2d, %2d)\n", i, _csr->representNode[i], _csr->ff[i], _csr->csrV[i]);
        #endif
    }
    #pragma endregion //PrepareNodeList

    /**
     * We've got two lists which are :
     * 1. d1Node_List       : _csr->degreeOneNodesQ->dataArr
     * 2. notD1Node_List    : _csr->notD1Node 
    */
    printf("Finished.\n");
    printf("D1 node count       (|D1|)              = %8d\n", _csr->foldedDegreeOneCount);
    printf("Hub node Count      (|Hub|)             = %8d\n", _csr->hubNodeCount);
    printf("Ordinary node Count (|V| - |D1|)        = %8d\n", _csr->ordinaryNodeCount);
    printf("==============================\n");
}