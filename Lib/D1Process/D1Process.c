#include "D1Process.h"

// #define DEBUG_D1

void D1Folding(struct CSR* _csr){
    struct qQueue* d1Q      = _csr->degreeOneNodesQ;

    int d1NodeID            = -1;
    int d1ParentID          = -1;
    int d1ParentNeighborID  = -1;
    
    int outerNodesNum       = 0;
    while(!qIsEmpty(d1Q)){
        d1NodeID                    = qPopFront(d1Q);
        _csr->nodesType[d1NodeID]   = D1;

        d1ParentID                  = _csr->csrE[_csr->csrV[d1NodeID]];
        _csr->nodesType[d1ParentID] = D1Hub;
        
        //移除d1Node時，去更新d1Nodes跟d1Parent的BC值
        outerNodesNum               = _csr->totalNodeNumber - _csr->representNode[d1NodeID];
        _csr->BCs[d1NodeID]         = _csr->BCs[d1NodeID] + (_csr->representNode[d1NodeID] - 1) * outerNodesNum;
        _csr->BCs[d1ParentID]       = _csr->BCs[d1ParentID] + (outerNodesNum - 1) * _csr->representNode[d1NodeID];
        //把壓縮的nodes數量累加進hub裡面
        _csr->representNode[d1ParentID] = _csr->representNode[d1ParentID] + _csr->representNode[d1NodeID];
        
        #ifdef DEBUG_D1
        printf("%d, linking to %d\n", d1NodeID, d1ParentID);
        #endif
        
        for(int d1ParentNeighborIndex = 0 ; d1ParentNeighborIndex < _csr->csrNodesDegree[d1ParentID] ; d1ParentNeighborIndex ++){
            d1ParentNeighborID = _csr->csrE[_csr->csrV[d1ParentID] + d1ParentNeighborIndex];
            if(d1ParentNeighborID == d1NodeID){
                swap(&(_csr->csrE[_csr->csrV[d1ParentID] + d1ParentNeighborIndex]), &(_csr->csrE[_csr->csrV[d1ParentID]]));
                break;
            }
        }
        // d1Node的offset不往後移動，為了能很快找到Parent
        // _csr->csrV[d1NodeID] ++;
        // d1Node的degree - 1
        // _csr->csrNodesDegree[d1NodeID] --;
        
        // d1Parent的offset往後移動一格，代表刪除d1Node
        _csr->csrV[d1ParentID] ++;
        // d1Parent的degree - 1;
        _csr->csrNodesDegree[d1ParentID] --;
        // 檢查parent是否也變成d1
        if(_csr->csrNodesDegree[d1ParentID] == 1){
            qPushBack(d1Q, d1ParentID);
        }
        
        //計數有多少D1 Nodes
        _csr->foldedDegreeOneCount ++;
    }
    
    int hubNodeCount = 0;
    for(int i = _csr->startNodeID ; i <= _csr->endNodeID ; i ++){
        if(_csr->nodesType[i] == D1Hub){
            hubNodeCount ++;
        }
        // printf("BC[%d] = %f\trepresent[%d] = %d\n", i, _csr->BCs[i], i, _csr->representNode[i]);
    }
    printf("hubNodeCount = %d\n", hubNodeCount);
}