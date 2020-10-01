#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

// Include constants
#include "settings.h"

// Kernels
#include "C_spmm_kernels.cu"

// Helper functions for matrix creation, conversion, prints
#include "C_spmm_helpers.cu"


int main()
{
	printf("Start of main()\n");

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

	if(DEBUG2){
    	printf("Before CudaMalloc\n");
    }

	float* d_sparse;
	float* d_val;
   	int* d_col_indx;
    int* d_row_indx;
    float* d_dense;
    float* d_dense_tr;
	float* d_out;
    cudaMalloc(&d_dense,  DROW * DCOL * sizeof(float));
	cudaMalloc(&d_out,    SROW * DCOL * sizeof(float));
	cudaMemcpy(d_dense,  dense,  DROW * DCOL * sizeof(float),
               cudaMemcpyHostToDevice);
	cudaMemcpy(d_out,    out,    SROW * DCOL * sizeof(float),
               cudaMemcpyHostToDevice);

    if(KERNEL == 1 || KERNEL == 23 || KERNEL == 26) {
    	if(DEBUG2){
    		printf("In KERNEL 1 CudaMalloc\n");
    	}
		cudaMalloc(&d_sparse, SROW * SCOL * sizeof(float));
		cudaMemcpy(d_sparse, sparse, SROW * SCOL * sizeof(float),
        	       cudaMemcpyHostToDevice);

		if(KERNEL == 26) {
		cudaMalloc(&d_dense_tr, SROW * SCOL * sizeof(float));
		dim3 dimBlock_tr(TRANSPOSEX, TRANSPOSEY);
   		dim3 dimGrid_tr(ceil( (float) SROW / dimBlock_tr.x), ceil( (float) DCOL / dimBlock_tr.y / WPT));
   		transpose<<<dimGrid_tr, dimBlock_tr>>>(DROW, DCOL, d_dense, d_dense_tr);
   			if(DEBUG1) {
   				float* dense_tr = (float *)malloc(DROW * DCOL * sizeof(float));
				for (int i=0; i<DROW*DCOL; i++) { out[i] = 0.0; }
   		    	cudaMemcpy(dense_tr,      d_dense_tr,  SROW * DCOL * sizeof(float),
               			   cudaMemcpyDeviceToHost);
               	printf("Dense_tr: \n"), print_matrix(DROW, DCOL, dense_tr);
				printf("\n");
   			}
		}       
    }
    // else if(KERNEL == 2 || KERNEL == 3) {
    else {
    	if(DEBUG2){
    		printf("In KERNEL 2 CudaMalloc\n");
    	}
		cudaMalloc(&d_val,      nnz * sizeof(float));
		cudaMalloc(&d_col_indx, nnz * sizeof(int));
		cudaMalloc(&d_row_indx, (SROW+1) * sizeof(int));
		if(DEBUG2){
    		printf("In KERNEL 2 CudaMemcpy\n");
    	}
		cudaMemcpy(d_val, 	   val,      nnz * sizeof(float),
        	       cudaMemcpyHostToDevice);
        if(DEBUG2) {
    		printf("CudaMemcopied d_val\n");
        }
       	cudaMemcpy(d_col_indx, col_indx, nnz * sizeof(int),
        	       cudaMemcpyHostToDevice);
        if(DEBUG2) {
    		printf("CudaMemcopied d_col_indx\n");
        }
       	cudaMemcpy(d_row_indx, row_indx, (SROW+1) * sizeof(int),
        	       cudaMemcpyHostToDevice);
        if(DEBUG2) {
    		printf("CudaMemcopied d_row_indxl\n");
        }
       	if(DEBUG2){
    		printf("After KERNEL 2 CudaMemcpy\n");
    	}
    }

    if(DEBUG2){
    	printf("After CudaMemcpy\n");
    }


    // Start the timed loop
    printf(">>> Starting %d MatMulKernel (Version %d) runs...\n", NUM_RUNS, KERNEL);
    gettimeofday(&Tvalue, &dummy);
    double starttime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);
    for (int r=0; r<NUM_RUNS; r++) {
    	if(KERNEL == 1) {
    	    // Invoke kernel
    		dim3 dimBlock(TS, TS);
   			dim3 dimGrid(ceil( (float) DCOL / dimBlock.x), ceil( (float) SROW /dimBlock.y));
    		if(1) {
    			printf("DimBlockX: %d, DimBLockY: %d\n", dimBlock.x, dimBlock.y);
    			printf("DimGridX: %d, DimGridY: %d\n", dimGrid.x, dimGrid.y);
    			printf("\n");
    		}
    		MatMulKernel<<<dimGrid, dimBlock>>>(d_sparse, d_dense, d_out, 
    											SCOL, DCOL, SROW);
    	}
    	else if(KERNEL == 2) {
    		// Invoke kernel
    		dim3 dimBlock(TS,TS);
   			dim3 dimGrid(ceil( (float) DCOL / dimBlock.x), ceil( (float) SROW /dimBlock.y));
    		if(1) {
    			printf("DimBlockX: %d, DimBLockY: %d\n", dimBlock.x, dimBlock.y);
    			printf("DimGridX: %d, DimGridY: %d\n", dimGrid.x, dimGrid.y);
    			printf("\n");
    		}
    		SparseMatMulKernel<<<dimGrid, dimBlock>>>(d_val, d_col_indx, d_row_indx, 
    												  d_dense, d_out, SCOL, DCOL, SROW);
    	}
    	else if(KERNEL == 3) {
    		// Invoke kernel
    		dim3 dimBlock(TS/WPT, TS);
   			dim3 dimGrid(ceil( (float) DCOL / dimBlock.x / WPT), ceil( (float) SROW / dimBlock.y));
    		if(1) {
    			printf("DimBlockX: %d, DimBLockY: %d\n", dimBlock.x, dimBlock.y);
    			printf("DimGridX: %d, DimGridY: %d\n", dimGrid.x, dimGrid.y);
    			printf("\n");
    		}
    		SparseMatMulKernel2<<<dimGrid, dimBlock>>>(d_val, d_col_indx, d_row_indx, 
    												   d_dense, d_out, SCOL, DCOL, SROW);
    	}
    	else if(KERNEL == 23) {
    		dim3 dimBlock(TS, TS/WPT);
   			dim3 dimGrid(ceil( (float) SROW / dimBlock.x), ceil( (float) DCOL / dimBlock.y / WPT));
   			if(1) {
    			printf("DimBlockX: %d, DimBLockY: %d\n", dimBlock.x, dimBlock.y);
    			printf("DimGridX: %d, DimGridY: %d\n", dimGrid.x, dimGrid.y);
    			printf("\n");
    		}
   			myGEMM3<<<dimGrid, dimBlock>>>(SROW, DCOL, SHARED, d_sparse, d_dense, d_out);
    	}
    	else if(KERNEL == 26) {
    		dim3 dimBlock(TSM/WPTM, TSN/WPTN);
   			dim3 dimGrid(ceil( (float) SROW / dimBlock.x / WPTM), ceil( (float) DCOL / dimBlock.y / WPT /WPTN));
   			if(1) {
    			printf("DimBlockX: %d, DimBLockY: %d\n", dimBlock.x, dimBlock.y);
    			printf("DimGridX: %d, DimGridY: %d\n", dimGrid.x, dimGrid.y);
    			printf("\n");
    		}
   			myGEMM6<<<dimGrid, dimBlock>>>(SROW, DCOL, SHARED, d_sparse, d_dense_tr, d_out);
   			
    	}
    	else if(KERNEL == 999) {
    	    int M = SROW;
    		int N = SHARED; 
    		int K = DCOL;
    		int B = DROW*DCOL / (N*K);

    		dim3 BLOCKS((32 * B * M + THREADS - 1) / THREADS, (K + 31) / 32);

    		if(1) {
    			printf("DimBlock: %d\n", THREADS);
    			printf("DimGridX: %d, DimGridY: %d\n", BLOCKS.x, BLOCKS.y);
    			printf("\n");
    		}

    		if(UseShflSync) {
    			spmm_kernel<<<BLOCKS, THREADS>>>(
            	d_row_indx, d_col_indx, d_val, d_dense, d_out,
            	B, M, N, K);
    		} 
    		else {
    			spmm_kernel_wo_shfl<<<BLOCKS, THREADS>>>(
            	d_row_indx, d_col_indx, d_val, d_dense, d_out,
            	B, M, N, K);
    		}
    		
    	}


    	cudaDeviceSynchronize();
    }

    // End the timed loop
    gettimeofday(&Tvalue, &dummy);
    double endtime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);
    double runtime = (endtime - starttime) / (double)NUM_RUNS;
    double flop = ((long)SCOL * (long)SROW * (long)DCOL * 2);

    // "theoretical" because for SpmmKernel does not account for sparsity and treats as full matrix
    printf(">>> Done: took %.7lf seconds per run, %.7lf (theoretical) GFLOPS\n", 
    		runtime, flop/runtime/(1000*1000*1000));

    // Read out from device memory
    cudaMemcpy(out,      d_out,  SROW * DCOL * sizeof(float),
               cudaMemcpyDeviceToHost);

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


	// Free memory
	if(KERNEL == 1 || KERNEL == 23 || KERNEL == 26)	{
		free(sparse);
		cudaFree(d_sparse);
	}
    // else if(KERNEL == 2 || KERNEL == 3 ) {
    else {
    	cudaFree(d_val);
    	cudaFree(d_col_indx);
    	cudaFree(d_row_indx);
    }
    free(dense);
    free(out);
    cudaFree(d_dense);
    cudaFree(d_out);
    if(DEBUG2) {
		printf("After CudaFree\n");
	}

	// Free host memory
	
    

	return 0;
}

