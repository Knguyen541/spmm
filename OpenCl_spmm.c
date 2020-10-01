#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

#include <CL/cl.h>

// Include constants
#include "settings.h"


// Helper functions for matrix creation, conversion, prints
#include "C_spmm_helpers.cu"

// Set the locations of the OpenCL kernel files
#define CL_INCLUDE_FILE "./settings.h"
#define CL_KERNEL_FILE "./OpenCL_spmm_kernels.cl"

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

}