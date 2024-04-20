# 编译器和编译选项
CC = gcc
CFLAGS = -O2

# 目标文件和可执行文件名
TARGET = a

# 指定库所在的目录
LIB_DIRS := ../Lib/
LIB_List := vVector qQueue tTime FileReader AdjList CSR D1Process AP_Process
LIB_True_Path := $(addprefix $(LIB_DIRS), $(LIB_List))

LIB_OBJS := $(addsuffix .o, $(LIB_List))

remain_OBJS := AP_Process.o

all: $(TARGET)

$(TARGET): $(LIB_OBJS)
	$(CC) $(CFLAGS) CC.c $^ -o $@

$(LIB_OBJS): %.o: $(LIB_DIRS)%
	$(CC) $(CFLAGS) -c $</$*.c -o $@

# # 清理生成的目标文件和可执行文件
clean:
	rm -f *.o $(TARGET)

# # 显示每个库对应的目标文件列表
# show_objs:
#     @echo $(LIB_OBJS)