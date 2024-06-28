/**
 * @author XPPGX
 * @date 2023/07/15
*/
#ifndef COMMON
#define COMMON

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#endif

//Content
#ifndef QQueue
#define QQueue
struct qQueue{
    int* dataArr;
    int front;
    int rear;
    int size;
};

struct qQueue* InitqQueue();
void qInitResize(struct qQueue* _queue, int _size);
void qPushBack(struct qQueue* _queue, int _val);
int qSpaceFull(struct qQueue* _queue);
int qIsEmpty(struct qQueue* _queue);
void qShowAllElement(struct qQueue* _queue);
int qPopFront(struct qQueue* _queue);
#endif