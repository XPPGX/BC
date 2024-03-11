#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif


#ifndef ADJLIST
#error Need include "AdjList.h", pls add "vVector.h" into "headers.h"
#endif

#ifndef QQueue
#error Need include "qQueue.h", pls add "qQueue.h" into "headers.h"
#endif

#ifndef cCSR
#error Need include "CSR.h", pls add "CSR.h" into "headers.h"
#endif

#ifndef D1Process
#define D1Process

#define Ordinary    1
#define D1          2
#define D1Hub       3

void D1Folding(struct CSR* _csr);
#endif