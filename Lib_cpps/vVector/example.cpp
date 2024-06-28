#ifndef COMMON
#define COMMON

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#endif

#include "vVector.h"
#include "vVector.cpp"

int main(){
    struct vVector* vec = InitvVector();
    for(int i = 0 ; i < 11 ; i ++){
        vAppend(vec, i * 10);
    }
    vShowAllElement(vec);
    printf("\n\n");
    vShrink(vec);
    vShowAllElement(vec);
}