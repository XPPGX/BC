###################   C++ #######################
# 编译器和编译选项
CC = g++
NVCC = nvcc

CFLAGS = -O2
NVCCFLAGS = -O2

# 目标文件和可执行文件名
TARGET = a

# 指定库所在的目录
LIB_DIRS := ../Lib_cpps/
LIB_List := vVector qQueue tTime FileReader AdjList CSR D1Process AP_Process
LIB_True_Path := $(addprefix $(LIB_DIRS), $(LIB_List))

LIB_OBJS := $(addsuffix .o, $(LIB_List))

MAIN_CU = gpu_fan.cu
MAIN_CU_OBJ = gpu_fan.o

remain_OBJS := AP_Process.o

all: $(TARGET)

$(TARGET): $(LIB_OBJS) $(MAIN_CU_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(LIB_OBJS): %.o: $(LIB_DIRS)%
	$(CC) $(CFLAGS) -c $</$*.cpp -o $@

$(MAIN_CU_OBJ): $(MAIN_CU)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# # 清理生成的目标文件和可执行文件
clean:
	rm -f *.o $(TARGET)
#################################################