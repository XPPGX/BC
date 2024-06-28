/**
 * @author XPPGX
 * @date 2023/07/15
*/
#include "FileReader.h"

//#define _DEBUG_

void getRowData(char* _row, int* _val1, int* _val2){
    char* token;
    
    token = strtok(_row, " ");
    *_val1 = atoi(token);

    token = strtok(NULL, " ");
    *_val2 = atoi(token);

    #ifdef _DEBUG_
    printf("node1 = %d, node2 = %d\n", *_val1, *_val2);
    #endif
}