/**
 * @author XPPGX
 * @date 2023/07/15
*/
#include "vVector.h"

//#define _DEBUG_
#ifdef VVECTOR
struct vVector* InitvVector(){
    struct vVector* vector = (struct vVector*)malloc(sizeof(struct vVector));
    vector->size = 5;
    vector->dataArr = (int*)malloc(sizeof(int) * 5);
    vector->tail = -1;

    #ifdef _DEBUG_
    printf("vector : size = %d, tail = %d, dataArr = %p\n", vector->size, vector->tail, vector->dataArr);
    #endif

    return vector;
}

void vShowAllElement(struct vVector* _vec){
    for(int i = 0 ; i <= _vec->tail ; i ++){
        printf("vec[%d] = %d\n", i, _vec->dataArr[i]);
    }
}

void vAppend(struct vVector* _vec, int _val){
    if(vIsFull(_vec)){

        #ifdef _DEBUG_
        printf("The Vector is full\n");
        #endif
        
        int* bigArray = (int*)malloc(sizeof(int) * (_vec->size) * 2);
        memcpy(bigArray, _vec->dataArr, sizeof(int) * (_vec->size));
        _vec->size = _vec->size * 2;
        free(_vec->dataArr);
        _vec->dataArr = bigArray;
    }
    _vec->tail ++;
    _vec->dataArr[_vec->tail] = _val;
    // printf("\tvec.tail = %d, vec.dataArr[%d] = %d\n", _vec->tail, _vec->tail, _vec->dataArr[_vec->tail]);
    
    #ifdef _DEBUG_
    printf("vec->dataArr[%d] = %d, append success\n", _vec->tail, _vec->dataArr[_vec->tail]);
    #endif
}

int vIsFull(struct vVector* _vec){
    if((_vec->tail + 1) == _vec->size){
        return 1;
    }
    else{
        return 0;
    }
}

void vShrink(struct vVector* _vec){
    #ifdef _DEBUG_
    printf("before shrink : vec->size = %d, vec->tail = %d\n", _vec->size, _vec->tail);
    #endif
    
    int* newSpace = (int*)malloc(sizeof(int) * (_vec->tail + 1));
    memcpy(newSpace, _vec->dataArr, sizeof(int) * (_vec->tail + 1));
    free(_vec->dataArr);
    _vec->dataArr = newSpace;
    _vec->size = _vec->tail + 1;

    #ifdef _DEBUG_
    printf("after shrink : vec->size = %d, vec->tail = %d\n", _vec->size, _vec->tail);
    #endif
}
#endif