/**
 * @author XPPGX
 * @brief This priority queue is made by binary heap
 * @date 2023/09/07
*/

// #define _DEBUG_

#include "PriorityQueue.h"

void minPQ_Init(struct minPQ* _minPQ_DATA){
    _minPQ_DATA->heapNodes = (struct Element*)calloc(6, sizeof(struct Element));
    _minPQ_DATA->heapNodes[0].element = -1; //第0格不用
    _minPQ_DATA->heapNodes[0].key = -1; //第0格不用
    
    _minPQ_DATA->elementIndexArr = (int*)calloc(6, sizeof(int));
    _minPQ_DATA->elementIndexArr[0] = -1; //第0格不用 

    _minPQ_DATA->size = 6;
    _minPQ_DATA->tail = 0;
}

int minPQ_IsFull(struct minPQ* _minPQ_DATA){
    if((_minPQ_DATA->tail + 1) == _minPQ_DATA->size){
        return 1;
    }
    else{
        return 0;
    }
}

void minPQ_swap(int* Val1, int* Val2){
    int temp = *Val1;
    *Val1 = *Val2;
    *Val2 = temp;
    // *Val1 = *Val1 ^ *Val2;

    // *Val2 = *Val1 ^ *Val2;
    
    // *Val1 = *Val1 ^ *Val2;
}

/**
 * @brief 這是由currentNodeIndex往下做的heapify，
 * 如果有element的key值增加，那
 * @param _currentNodeIndex 就是增加key值的那個element。
*/
void minPQ_minHeapify(struct minPQ* _minPQ_DATA, int _currentNodeIndex){
    int currentNodeIndex = _currentNodeIndex; //新插入的node的parent當成currentNodeIndex
    int leftChildNodeIndex;
    int rightChildNodeIndex;
    int smallestNodeIndex;

    // 不斷往上檢查到root。
    while(currentNodeIndex <= _minPQ_DATA->tail){ //只要current不是root就繼續while
        // 取得左子、右子的index
        leftChildNodeIndex = currentNodeIndex * 2;
        rightChildNodeIndex = leftChildNodeIndex + 1;
        // currentNode跟leftChildNode比
        if(leftChildNodeIndex <= _minPQ_DATA->tail && _minPQ_DATA->heapNodes[leftChildNodeIndex].key < _minPQ_DATA->heapNodes[currentNodeIndex].key){
            smallestNodeIndex = leftChildNodeIndex;
        }
        else{
            smallestNodeIndex = currentNodeIndex;
        }
        // currentNode跟rightChildNode比
        if(rightChildNodeIndex <= _minPQ_DATA->tail && _minPQ_DATA->heapNodes[rightChildNodeIndex].key < _minPQ_DATA->heapNodes[smallestNodeIndex].key){
            smallestNodeIndex = rightChildNodeIndex;
        }
        // 如果current不是最小的，則跟當前最小的交換
        if(smallestNodeIndex != currentNodeIndex){
            int smallestElement = _minPQ_DATA->heapNodes[smallestNodeIndex].element;
            int currentElement = _minPQ_DATA->heapNodes[currentNodeIndex].element;
            minPQ_swap(&(_minPQ_DATA->heapNodes[smallestNodeIndex].key), &(_minPQ_DATA->heapNodes[currentNodeIndex].key));
            minPQ_swap(&(_minPQ_DATA->heapNodes[smallestNodeIndex].element), &(_minPQ_DATA->heapNodes[currentNodeIndex].element));
            minPQ_swap(&(_minPQ_DATA->elementIndexArr[smallestElement]), &(_minPQ_DATA->elementIndexArr[currentElement]));
            currentNodeIndex = smallestNodeIndex;
        }
        else{ //如果current是最小的，則break
            break;
        }
    }
}

/**
 * @brief 插入element的時候，會從tail到root不斷做heapify(不斷檢查"自己"與"自己的parent")
*/
void minPQ_Insert(struct minPQ* _minPQ_DATA, int _newElement, int _newKey){
    // 若PQ滿了，則擴展空間
    if(minPQ_IsFull(_minPQ_DATA)){
        #ifdef _DEBUG_
        printf("The PQ is full\n");
        #endif

        struct Element* bigElementArr = (struct Element*)calloc((_minPQ_DATA->size ) * 2, sizeof(struct Element));
        memcpy(bigElementArr, _minPQ_DATA->heapNodes, sizeof(struct Element) * _minPQ_DATA->size);
        free(_minPQ_DATA->heapNodes);
        _minPQ_DATA->heapNodes = bigElementArr;

        int* bigElementIndexArr = (int*)calloc((_minPQ_DATA->size) * 2, sizeof(int));
        memcpy(bigElementIndexArr, _minPQ_DATA->elementIndexArr, sizeof(int) * _minPQ_DATA->size);
        free(_minPQ_DATA->elementIndexArr);
        _minPQ_DATA->elementIndexArr = bigElementIndexArr;

        _minPQ_DATA->size = _minPQ_DATA->size * 2;
    }
    // 在尾端塞入資料
    _minPQ_DATA->tail ++;
    _minPQ_DATA->heapNodes[_minPQ_DATA->tail].element = _newElement;
    _minPQ_DATA->heapNodes[_minPQ_DATA->tail].key = _newKey;
    _minPQ_DATA->elementIndexArr[_minPQ_DATA->tail] = _minPQ_DATA->tail;

    #ifdef _DEBUG_
    printf("minPQ_DATA->heapNodes[%d] => element = %d, key = %d, index = %d\n", _minPQ_DATA->tail, _minPQ_DATA->heapNodes[_minPQ_DATA->tail].element, _minPQ_DATA->heapNodes[_minPQ_DATA->tail].key, _minPQ_DATA->elementIndexArr[_minPQ_DATA->tail]);
    // printf("heapify...\n");
    #endif
    
    // 處理insertion case的heapify，從tail往root做heapify
    // 只要不斷比較"自己"與"自己的parent"就可以了
    int currentNodeIndex = _minPQ_DATA->tail;
    int parentNodeIndex;
    while(1){
        parentNodeIndex = currentNodeIndex / 2;
        if(parentNodeIndex >= 1 && _minPQ_DATA->heapNodes[currentNodeIndex].key < _minPQ_DATA->heapNodes[parentNodeIndex].key){
            int currentElement = _minPQ_DATA->heapNodes[currentNodeIndex].element;
            int parentElement = _minPQ_DATA->heapNodes[parentNodeIndex].element;

            #ifdef _DEBUG_
            printf("current = {element = %d, key = %d, index = %d}, parent = {element = %d, key = %d, index = %d}, swap both!\n", currentElement, _minPQ_DATA->heapNodes[currentNodeIndex].key, _minPQ_DATA->elementIndexArr[currentElement], parentElement, _minPQ_DATA->heapNodes[parentNodeIndex].key, _minPQ_DATA->elementIndexArr[parentElement]);
            #endif

            // 交換element與key
            minPQ_swap(&(_minPQ_DATA->heapNodes[currentNodeIndex].element), &(_minPQ_DATA->heapNodes[parentNodeIndex].element));
            minPQ_swap(&(_minPQ_DATA->heapNodes[currentNodeIndex].key), &(_minPQ_DATA->heapNodes[parentNodeIndex].key));
            // 交換element的index
            minPQ_swap(&(_minPQ_DATA->elementIndexArr[currentElement]), &(_minPQ_DATA->elementIndexArr[parentElement]));

            #ifdef _DEBUG_
            printf("current = {element = %d, key = %d, index = %d}, parent = {element = %d, key = %d, index = %d}, swap finished\n", _minPQ_DATA->heapNodes[currentNodeIndex].element, _minPQ_DATA->heapNodes[currentNodeIndex].key, _minPQ_DATA->elementIndexArr[currentElement], _minPQ_DATA->heapNodes[parentNodeIndex].element, _minPQ_DATA->heapNodes[parentNodeIndex].key, _minPQ_DATA->elementIndexArr[parentElement]);
            #endif
            // current往heap上移一層
            currentNodeIndex = parentNodeIndex;
        }
        else{
            break;
        }
    }
}

void minPQ_increaseKey(struct minPQ* _minPQ_DATA, int _element, int _increaseVal){
    // 取得element在min heap中的index
    int index = _minPQ_DATA->elementIndexArr[_element];
    // 把key值增加_increaseVal
    _minPQ_DATA->heapNodes[index].key = _minPQ_DATA->heapNodes[index].key + _increaseVal;
    minPQ_minHeapify(_minPQ_DATA, index);
}

/**
 * @brief return位於min heap頂端的element
*/
int minPQ_minValueElement(struct minPQ* _minPQ_DATA){
    return _minPQ_DATA->heapNodes[1].element;
}

/**
 * @brief return位於min heap頂端的Key
*/
int minPQ_minValue(struct minPQ* _minPQ_DATA){
    return _minPQ_DATA->heapNodes[1].key;
}

/**
 * @brief return Min heap中總共有幾個element
*/
int minPQ_size(struct minPQ* _minPQ_DATA){
    return _minPQ_DATA->tail;
}