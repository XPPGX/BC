/**
 * @author XPPGX
 * @date 2023/07/22
*/
#include "tTime.h"

extern inline double seconds(){
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}