
//清除該行並把游標放回該 row 開頭
#define PCTL_clear_line() printf("\033[K\r")




//游標回到上一 row       
#define PCTL_goto_previous_line() printf("\033[1A")



