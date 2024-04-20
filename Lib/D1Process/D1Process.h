#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif


#ifndef ADJLIST
// #error Need include "AdjList.h", pls add "vVector.h" into "headers.h"
#include "../AdjList/AdjList.h"
#endif

#ifndef QQueue
// #error Need include "qQueue.h", pls add "qQueue.h" into "headers.h"
#include "../qQueue/qQueue.h"
#endif

#ifndef cCSR
// #error Need include "CSR.h", pls add "CSR.h" into "headers.h"
#include "../CSR/CSR.h"
#endif

#ifndef D1Process
#define D1Process

void D1Folding(struct CSR* _csr);
#endif