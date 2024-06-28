#ifndef DP
#define DP
#include "DebugPrintf.h"
#endif

#ifdef _DEBUG_
extern inline void DebugLog(const char* format, ...){
    va_list argPtr;
    int count;
    va_start(argPtr, format);
    fflush(stderr);
    count = vfprintf(stderr, format, argPtr);
    va_end(argPtr);
}
#else
extern inline void DebugLog(const char* format, ...){

}
#endif