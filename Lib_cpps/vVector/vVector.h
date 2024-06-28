/**
 * @author XPPGX
 * @date 2023/07/15
*/
#ifndef COMMON
#define COMMON

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#endif

//Content
#ifndef VVECTOR
#define VVECTOR
struct vVector{
    int* dataArr;
    int tail;
    int size;
};


struct vVector* InitvVector();
void vShowAllElement(struct vVector* _vec);
void vAppend(struct vVector* _vec, int _val);
int vIsFull(struct vVector* _vec);
void vShrink(struct vVector* _vec);
#endif