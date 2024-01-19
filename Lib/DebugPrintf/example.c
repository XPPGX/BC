#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif


#ifndef DP
#define DP
#include "DebugPrintf.h"
#endif



int main(int argc, char* argv[]){
    int a1 = atoi(argv[1]);
    int a2 = atoi(argv[2]);
    DebugLog("1st = %d, 2nd = %d\n", a1, a2);
}