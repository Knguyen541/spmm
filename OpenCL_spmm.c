#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>

#include <CL/cl.h>

// Include constants
#include "settings.h"

// Kernels
// #include "OpenCL_spmm_kernels.cl"

// Helper functions for matrix creation, conversion, prints
#include "C_spmm_helpers.cu"

// Set the locations of the OpenCL kernel files
#define CL_INCLUDE_FILE "./settings.h"
#define CL_KERNEL_FILE "./OpenCL_spmm_kernels.cl"

// Forward declaration of the OpenCL error checking function
void checkError(cl_int error, int line);

// =================================================================================================

// Load an OpenCL kernel from file
char* readKernelFile(const char* filename, long* _size) {

    // Open the file
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("-- Error opening file %s\n", filename);
        exit(1);
    }

    // Get its size
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);

    // Read the kernel code as a string
    char* source = (char *)malloc((size+1)*sizeof(char));
    fread(source, 1, size*sizeof(char), file);
    source[size] = '\0';
    fclose(file);

    // Save the size and return the source string
    *_size = (size+1);
    return source;
}

// =================================================================================================

int main()
{
    int srow = SROW;
    int scol = SCOL;
    int drow = DROW;
    int dcol = DCOL; 

	// Define OpenCL variables
    cl_int err;

	// Timers
    struct timeval Tvalue;
    struct timezone dummy;

    float *sparse = (float *)malloc(SROW * SCOL * sizeof(float));
	create_random_sparse_matrix(SROW, SCOL, SPARSITY, sparse);
	if(DEBUG1) {
		printf("Sparse Matrix: \n"), print_matrix(SROW, SCOL, sparse);
	}
	
	if(DEBUG2){
    	printf("Created Sparse Matrix\n");
    }

	int nnz = count_nnz(SROW, SCOL, sparse);

	if(DEBUG2){
    	printf("Exited Count_NNZ\n");
    	printf("NNZ: %d\n", nnz);
    }
	
	float val[nnz];
	if(DEBUG2){
    	printf("Created Val Array\n");
    }
	int col_indx[nnz];
	if(DEBUG2){
    	printf("Created Col_Indx Array\n");
    }
	int row_indx[SROW+1];
	if(DEBUG2){
    	printf("Created Row_Indx Array\n");
    }
	if(DEBUG2){
    	printf("Calling Conversion of Sparse to CSR\n");
    }
	dense_to_csr(nnz, SROW, SCOL, sparse, val, col_indx, row_indx);
	if(DEBUG1) {
		printf("VAL: "), print_float_array(nnz, val);
		printf("COL: "), print_int_array(nnz, col_indx);
		printf("ROW: "), print_int_array(SROW+1, row_indx);
		printf("\n");
	}

	if(DEBUG2){
    	printf("Created CSR representation of Sparse Matrix\n");
    }

	float *dense = (float *)malloc(DROW * DCOL * sizeof(float));
	create_random_matrix(DROW, DCOL, dense);
	if(DEBUG1) {
		printf("Dense Matrix: \n"), print_matrix(DROW, DCOL, dense);
		printf("\n");
	}

	if(DEBUG2){
    	printf("Created Dense Matrix\n");
    }

	float* out = (float *)malloc(SROW * DCOL * sizeof(float));
	for (int i=0; i<SROW*DCOL; i++) { out[i] = 0.0; }
	if(DEBUG1) {
		printf("Out Matrix Init: \n"), print_matrix(SROW, DCOL, out);
		printf("\n");
	}

	// Configure the OpenCL environment
    printf(">>> Initializing OpenCL...\n");
    cl_platform_id platform = 0;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
    char deviceName[1024];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 1024, deviceName, NULL);
    cl_event event = NULL;

    // Read the kernel file from disk
    long sizeHeader, sizeSource;
    char* header = readKernelFile(CL_INCLUDE_FILE, &sizeHeader);
    char* source = readKernelFile(CL_KERNEL_FILE, &sizeSource);
    long size = 2 + sizeHeader + sizeSource;
    char* code = (char*)malloc(size*sizeof(char));
    for (int c=0; c<size; c++) { code[c] = '\0'; }
    strcat(code, header);
    strcat(code, source);
    const char* constCode = code;
    free(header);
    free(source);

    cl_program program = clCreateProgramWithSource(context, 1, &constCode, NULL, NULL);
    clBuildProgram(program, 0, NULL, "", NULL, NULL);

	// Check for compilation errors
    size_t logSize;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    char* messages = (char*)malloc((1+logSize)*sizeof(char));
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, messages, NULL);
    messages[logSize] = '\0';
    if (logSize > 10) { printf(">>> Compiler message: %s\n", messages); }
    free(messages);

    // Prepare OpenCL memory objects
    cl_mem buf_sparse = clCreateBuffer(context, CL_MEM_READ_ONLY,  SROW*SCOL*sizeof(float), NULL, NULL);
    cl_mem buf_dense  = clCreateBuffer(context, CL_MEM_READ_ONLY,  DROW*DCOL*sizeof(float), NULL, NULL);
    cl_mem buf_out    = clCreateBuffer(context, CL_MEM_READ_WRITE, SROW*DCOL*sizeof(float), NULL, NULL);

    cl_mem buf_val      = clCreateBuffer(context, CL_MEM_READ_ONLY,  nnz*sizeof(float),     NULL, NULL);
    cl_mem buf_col_indx = clCreateBuffer(context, CL_MEM_READ_ONLY,  nnz*sizeof(int),       NULL, NULL);
    cl_mem buf_row_indx = clCreateBuffer(context, CL_MEM_READ_ONLY,  (SROW+1)*sizeof(int),  NULL, NULL);


    // Copy matrices to the GPU
    clEnqueueWriteBuffer(queue, buf_sparse, CL_TRUE, 0, SROW*SCOL*sizeof(float), sparse, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_dense,  CL_TRUE, 0, DROW*DCOL*sizeof(float), dense,  0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_out,    CL_TRUE, 0, SROW*DCOL*sizeof(float), out,    0, NULL, NULL);

    clEnqueueWriteBuffer(queue, buf_val,      CL_TRUE, 0, nnz*sizeof(float),      val, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_col_indx, CL_TRUE, 0, nnz*sizeof(float),      col_indx,  0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_row_indx, CL_TRUE, 0, (SROW+1)*sizeof(float), row_indx,    0, NULL, NULL);

    // Configure the myGEMM kernel and set its arguments
    char kernelname[100];
    sprintf(kernelname, "spmm_kernel%d", KERNEL);
    cl_kernel kernel1 = clCreateKernel(program, kernelname, NULL);

    #if KERNEL == 1
        err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void*)&buf_sparse);
        err = clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void*)&buf_dense);
        err = clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void*)&buf_out);
        err = clSetKernelArg(kernel1, 3, sizeof(int), (void*)&scol);
        err = clSetKernelArg(kernel1, 4, sizeof(int), (void*)&dcol);
        err = clSetKernelArg(kernel1, 5, sizeof(int), (void*)&srow);
    #else
        err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void*)&buf_val);
        err = clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void*)&buf_col_indx);
        err = clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void*)&buf_row_indx);
        err = clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void*)&buf_dense);
        err = clSetKernelArg(kernel1, 4, sizeof(cl_mem), (void*)&buf_out);
        err = clSetKernelArg(kernel1, 5, sizeof(int), (void*)&scol);
        err = clSetKernelArg(kernel1, 6, sizeof(int), (void*)&dcol);
        err = clSetKernelArg(kernel1, 7, sizeof(int), (void*)&srow);
    #endif
    checkError(err,__LINE__);

    #if KERNEL == 1 || KERNEL == 2
        const size_t local[2] = { TS, TS };
        const size_t global[2] = { (size_t)DCOL, (size_t)SROW };
    #elif KERNEL == 3
        const size_t local[2] = { TS/WPT, TS };
        const size_t global[2] = { (size_t)DCOL/WPT, (size_t)SROW };
    #endif
    // Start the timed loop
    printf(">>> Starting %d myGEMM%d runs...\n", NUM_RUNS, KERNEL);
    gettimeofday(&Tvalue, &dummy);
    double starttime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);
    for (int r=0; r<NUM_RUNS; r++) {

        // Run the transpose kernel first
        //#if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8 || KERNEL == 9 || KERNEL == 10
        //    err = clEnqueueNDRangeKernel(queue, kernel2, 2, NULL, tGlobal, tLocal, 0, NULL, &event);
        //#endif

        err = clEnqueueNDRangeKernel(queue, kernel1, 2, NULL, global, local, 0, NULL, &event);

        // Wait for calculations to be finished
        checkError(err,__LINE__);
        err = clWaitForEvents(1, &event);
    }

    // End the timed loop
    gettimeofday(&Tvalue, &dummy);
    double endtime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);
    double runtime = (endtime - starttime) / (double)NUM_RUNS;
    double flop = ((long)SCOL * (long)SROW * (long)DCOL * 2);

    // "theoretical" because for SpmmKernel does not account for sparsity and treats as full matrix
    printf(">>> Done: took %.7lf seconds per run, %.7lf (theoretical) GFLOPS\n", 
            runtime, flop/runtime/(1000*1000*1000));

    // Copy the output matrix C back to the CPU memory
    clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, SROW*DCOL*sizeof(float), out, 0, NULL, NULL);

    if(DEBUG1) {
        printf("Out Matrix: \n"), print_matrix(SROW, DCOL, out);
    }


    if(DEBUG2) {
        printf("Checking Out Matrix: \n");
    }

    if (CHECK && check_mm(sparse, dense, out, SROW, SCOL, DCOL) == 1) {
        printf("Correct Result! \n");
    }
    else if(CHECK) {
        printf("Wrong Result! \n");
    }


    // Free the OpenCL memory objects
    clReleaseMemObject(buf_sparse);
    clReleaseMemObject(buf_dense);
    //clReleaseMemObject(bufB_TR);
    clReleaseMemObject(buf_out);

    // Clean-up OpenCL 
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseKernel(kernel1);

    // Free the host memory objects
    free(sparse);
    free(dense);
    free(out);
}

// =================================================================================================

// Print an error message to screen (only if it occurs)
void checkError(cl_int error, int line) {
    if (error != CL_SUCCESS) {
        switch (error) {
            case CL_DEVICE_NOT_FOUND:                 printf("-- Error at %d:  Device not found.\n", line); break;
            case CL_DEVICE_NOT_AVAILABLE:             printf("-- Error at %d:  Device not available\n", line); break;
            case CL_COMPILER_NOT_AVAILABLE:           printf("-- Error at %d:  Compiler not available\n", line); break;
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:    printf("-- Error at %d:  Memory object allocation failure\n", line); break;
            case CL_OUT_OF_RESOURCES:                 printf("-- Error at %d:  Out of resources\n", line); break;
            case CL_OUT_OF_HOST_MEMORY:               printf("-- Error at %d:  Out of host memory\n", line); break;
            case CL_PROFILING_INFO_NOT_AVAILABLE:     printf("-- Error at %d:  Profiling information not available\n", line); break;
            case CL_MEM_COPY_OVERLAP:                 printf("-- Error at %d:  Memory copy overlap\n", line); break;
            case CL_IMAGE_FORMAT_MISMATCH:            printf("-- Error at %d:  Image format mismatch\n", line); break;
            case CL_IMAGE_FORMAT_NOT_SUPPORTED:       printf("-- Error at %d:  Image format not supported\n", line); break;
            case CL_BUILD_PROGRAM_FAILURE:            printf("-- Error at %d:  Program build failure\n", line); break;
            case CL_MAP_FAILURE:                      printf("-- Error at %d:  Map failure\n", line); break;
            case CL_INVALID_VALUE:                    printf("-- Error at %d:  Invalid value\n", line); break;
            case CL_INVALID_DEVICE_TYPE:              printf("-- Error at %d:  Invalid device type\n", line); break;
            case CL_INVALID_PLATFORM:                 printf("-- Error at %d:  Invalid platform\n", line); break;
            case CL_INVALID_DEVICE:                   printf("-- Error at %d:  Invalid device\n", line); break;
            case CL_INVALID_CONTEXT:                  printf("-- Error at %d:  Invalid context\n", line); break;
            case CL_INVALID_QUEUE_PROPERTIES:         printf("-- Error at %d:  Invalid queue properties\n", line); break;
            case CL_INVALID_COMMAND_QUEUE:            printf("-- Error at %d:  Invalid command queue\n", line); break;
            case CL_INVALID_HOST_PTR:                 printf("-- Error at %d:  Invalid host pointer\n", line); break;
            case CL_INVALID_MEM_OBJECT:               printf("-- Error at %d:  Invalid memory object\n", line); break;
            case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  printf("-- Error at %d:  Invalid image format descriptor\n", line); break;
            case CL_INVALID_IMAGE_SIZE:               printf("-- Error at %d:  Invalid image size\n", line); break;
            case CL_INVALID_SAMPLER:                  printf("-- Error at %d:  Invalid sampler\n", line); break;
            case CL_INVALID_BINARY:                   printf("-- Error at %d:  Invalid binary\n", line); break;
            case CL_INVALID_BUILD_OPTIONS:            printf("-- Error at %d:  Invalid build options\n", line); break;
            case CL_INVALID_PROGRAM:                  printf("-- Error at %d:  Invalid program\n", line); break;
            case CL_INVALID_PROGRAM_EXECUTABLE:       printf("-- Error at %d:  Invalid program executable\n", line); break;
            case CL_INVALID_KERNEL_NAME:              printf("-- Error at %d:  Invalid kernel name\n", line); break;
            case CL_INVALID_KERNEL_DEFINITION:        printf("-- Error at %d:  Invalid kernel definition\n", line); break;
            case CL_INVALID_KERNEL:                   printf("-- Error at %d:  Invalid kernel\n", line); break;
            case CL_INVALID_ARG_INDEX:                printf("-- Error at %d:  Invalid argument index\n", line); break;
            case CL_INVALID_ARG_VALUE:                printf("-- Error at %d:  Invalid argument value\n", line); break;
            case CL_INVALID_ARG_SIZE:                 printf("-- Error at %d:  Invalid argument size\n", line); break;
            case CL_INVALID_KERNEL_ARGS:              printf("-- Error at %d:  Invalid kernel arguments\n", line); break;
            case CL_INVALID_WORK_DIMENSION:           printf("-- Error at %d:  Invalid work dimensionsension\n", line); break;
            case CL_INVALID_WORK_GROUP_SIZE:          printf("-- Error at %d:  Invalid work group size\n", line); break;
            case CL_INVALID_WORK_ITEM_SIZE:           printf("-- Error at %d:  Invalid work item size\n", line); break;
            case CL_INVALID_GLOBAL_OFFSET:            printf("-- Error at %d:  Invalid global offset\n", line); break;
            case CL_INVALID_EVENT_WAIT_LIST:          printf("-- Error at %d:  Invalid event wait list\n", line); break;
            case CL_INVALID_EVENT:                    printf("-- Error at %d:  Invalid event\n", line); break;
            case CL_INVALID_OPERATION:                printf("-- Error at %d:  Invalid operation\n", line); break;
            case CL_INVALID_GL_OBJECT:                printf("-- Error at %d:  Invalid OpenGL object\n", line); break;
            case CL_INVALID_BUFFER_SIZE:              printf("-- Error at %d:  Invalid buffer size\n", line); break;
            case CL_INVALID_MIP_LEVEL:                printf("-- Error at %d:  Invalid mip-map level\n", line); break;
            case -1024:                               printf("-- Error at %d:  *clBLAS* Functionality is not implemented\n", line); break;
            case -1023:                               printf("-- Error at %d:  *clBLAS* Library is not initialized yet\n", line); break;
            case -1022:                               printf("-- Error at %d:  *clBLAS* Matrix A is not a valid memory object\n", line); break;
            case -1021:                               printf("-- Error at %d:  *clBLAS* Matrix B is not a valid memory object\n", line); break;
            case -1020:                               printf("-- Error at %d:  *clBLAS* Matrix C is not a valid memory object\n", line); break;
            case -1019:                               printf("-- Error at %d:  *clBLAS* Vector X is not a valid memory object\n", line); break;
            case -1018:                               printf("-- Error at %d:  *clBLAS* Vector Y is not a valid memory object\n", line); break;
            case -1017:                               printf("-- Error at %d:  *clBLAS* An input dimension (M,N,K) is invalid\n", line); break;
            case -1016:                               printf("-- Error at %d:  *clBLAS* Leading dimension A must not be less than the size of the first dimension\n", line); break;
            case -1015:                               printf("-- Error at %d:  *clBLAS* Leading dimension B must not be less than the size of the second dimension\n", line); break;
            case -1014:                               printf("-- Error at %d:  *clBLAS* Leading dimension C must not be less than the size of the third dimension\n", line); break;
            case -1013:                               printf("-- Error at %d:  *clBLAS* The increment for a vector X must not be 0\n", line); break;
            case -1012:                               printf("-- Error at %d:  *clBLAS* The increment for a vector Y must not be 0\n", line); break;
            case -1011:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix A is too small\n", line); break;
            case -1010:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix B is too small\n", line); break;
            case -1009:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix C is too small\n", line); break;
            case -1008:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector X is too small\n", line); break;
            case -1007:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector Y is too small\n", line); break;
            case -1001:                               printf("-- Error at %d:  Code -1001: no GPU available?\n", line); break;
            default:                                  printf("-- Error at %d:  Unknown with code %d\n", line, error);
        }
        exit(1);
    }
}

// =================================================================================================