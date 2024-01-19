#include "qQueue.h"
#include "qQueue.c"
int main(){
    struct qQueue* queue = InitqQueue();
    printf("dataArr.addr = %p, queueSize = %d\n", queue->dataArr, queue->size);
    qInitResize(queue, 20);
    printf("dataArr.addr = %p, queueSize = %d\n", queue->dataArr, queue->size);
    for(int i = 0 ; i < 10 ; i ++){
        qPushBack(queue, i * 10);
    }
    qShowAllElement(queue);
    for(int i = 0 ; i < 10 ; i ++){
        printf("Q[%d] = %d\n", i, qPopFront(queue));
    }
    printf("queue->size = %d, queue->front = %d, queue->rear = %d\n", queue->size, queue->front, queue->rear);
    int x = qPopFront(queue);
    x = qPopFront(queue);
    x = qPopFront(queue);
    for(int i = 0 ; i < 10 ; i ++){
        qPushBack(queue, i * 20);
    }
    for(int i = 0 ; i < 10 ; i ++){
        qPushBack(queue, i * 20);
    }
    for(int i = 0 ; i < 10 ; i ++){
        qPushBack(queue, i * 20);
    }
    printf("queue->size = %d, queue->front = %d, queue->rear = %d\n", queue->size, queue->front, queue->rear);
    qShowAllElement(queue);
}