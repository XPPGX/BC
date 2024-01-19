#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif


#ifndef ADJLIST
#include "../AdjList/AdjList.h"
// #error Need include "AdjList.h", pls add "vVector.h" into "headers.h"
#endif

#ifndef QQueue
#include "../qQueue/qQueue.h"
// #error Need include "qQueue.h", pls add "qQueue.h" into "headers.h"
#endif

#ifndef cCSR
#include "../CSR/CSR.h"
// #error Need include "CSR.h", pls add "CSR.h" into "headers.h"
#endif

#ifndef D1Process
#define D1Process

void D1Folding(struct CSR* _csr);
#endif