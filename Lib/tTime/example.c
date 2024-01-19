#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

//content
#include "tTime.h"
#include "tTime.c"

int main(){
    double startTime = seconds();
    for(int i = 0 ; i < 100000000 ; i ++){
        int x = 1;
    }
    double endTime = seconds();
    printf("execution time = %f\n", endTime - startTime);
}