/**
 * @author XPPGX
 * @brief This min priority queue is made by binary heap
 * @date 2023/09/07
*/
#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

struct Element{
    int element; //color class
    int key; //the number of node of color class
};


struct minPQ{
    struct Element* heapNodes;
    int* elementIndexArr;
    int tail;
    int size;
};

/**
 * @brief 以min binary heap實作，並以index = 1當作root。
*/
void minPQ_Init(struct minPQ* _minPQ_DATA);

int minPQ_IsFull(struct minPQ* _minPQ_DATA);
void minPQ_swap(int* Val1, int* Val2);
void minPQ_minHeapify(struct minPQ* _minPQ_DATA, int _currentNodeIndex);
void minPQ_Insert(struct minPQ* _minPQ_DATA, int _newElement, int _newKey);
void minPQ_increaseKey(struct minPQ* _minPQ_DATA, int _element, int _increaseVal);
int minPQ_minValueElement(struct minPQ* _minPQ_DATA);
int minPQ_minValue(struct minPQ* _minPQ_DATA);
int minPQ_size(struct minPQ* _minPQ_DATA);