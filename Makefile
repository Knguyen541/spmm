CC = gcc
NVCC =/usr/local/cuda/bin/nvcc
CFLAGS = -Wall -Wextra 
OPENCL_LIB_PATH = -I/usr/local/cuda-10.2/targets/x86_64-linux/include -L/usr/local/cuda-10.2/targets/x86_64-linux/lib -lOpenCL


.PHONY: all clean

defalt: cuda

cuda:
	$(NVCC) -arch=sm_35 -o C_spmm C_spmm.cu 

run:
	./C_spmm

new: clean cuda run



opencl:
	$(CC) -o OpenCL_spmm OpenCL_spmm.c $(OPENCL_LIB_PATH)

run_cl:
	./OpenCL_spmm

new_cl: clean opencl run_cl



clean:
	rm -f C_spmm OpenCL_spmm

hello:
	$(CC) -o Hello cudaHello.cu 