// https://www.tutorialspoint.com/cuda/cuda_matrix_multiplication.htm
__global__ void MatMulKernel(float* d_M, float* d_N, float* d_P, 
							 int width, int width2, int out_row)
{
	int out_col = width2;

	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;

	if(DEBUG){
		printf("Thread [%d,%d,%d,%d,%d] \
        		From device\n", threadIdx.x,threadIdx.y,blockIdx.x,row,col);
	}

	if(row<out_row&& col <out_col) {
   		float product_val = 0;
   		for(int k=0;k<width;k++) {
      		product_val += d_M[row*width+k]*d_N[k*width2+col];
      		if(row == 4 && col == 2 && DEBUG) {
      			printf("Producval from thread [%d,%d,%d,%f,%d,%f,%f] \
        				From device\n", threadIdx.x,threadIdx.y,blockIdx.x, \
        				product_val, k, d_M[row*width+k], d_N[k*width2+col]);
      		}
   		}
   		d_P[row*width2+col] = product_val;
	}
}

// =================================================================================================

// https://on-demand.gputechconf.com/gtc/2012/presentations/S0285-Optimization-of-Sparse-Matrix-Matrix-Multiplication-on-GPU.pdf
__global__ void SparseMatMulKernel(float* val, int* col_indx, int* row_indx, float* dense, 
								   float* out, int width, int width2, int out_row)
{
	int out_col = width2;

	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;

	if(row<out_row&& col <out_col) {
		float product_val = 0;
		int row_start = row_indx[row];
		int row_end = row_indx[row+1];
		if(DEBUG) {
      		printf("Row: %d, Col: %d, row_start: %d, row_end: %d] \
        			From device\n", row, col, row_start, row_end);
      	}

		for(int k = row_start; k < row_end; k++) {
			product_val += val[k] * dense[col_indx[k]*width2 + col];
			if(row == 2 && col == 0 && DEBUG) {
      			printf("Producval from thread [%d,%d,%f,%d,%f,%f] \
        				From device\n", row, col, \
        				product_val, k, val[k], dense[col_indx[k]*width2 + col]);
      		}
		}
		out[row*width2+col] = product_val;
	}
}

// =================================================================================================

// !!! Incorrect result for SIZE = 8, but for 4,16,32,...works // Seems to work now
// Correct with WPT = 1; Also fastest with WPT = 1: ~700GFlops
// Increased the amount of work-per-thread by a factor WPT and use shared memory
__global__ void SparseMatMulKernel2(float* val, int* col_indx, int* row_indx, float* dense, 
								    float* out, int width, int width2, int out_row)
{
	int out_col = width2;

	int global_row = blockIdx.y*blockDim.y+threadIdx.y;
	int global_col = blockIdx.x*blockDim.x*WPT+threadIdx.x;

	if(DEBUG) {
      	printf("Row: %d, Col: %d, blockIdx.x: %d, blockIdx.y: %d, blockDim.x: %d, blockDim.y: %d, WPT: %d\n", 
      			global_row, global_col, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, WPT);
    }

	if(global_row < out_row && global_col < out_col) { //!!!
		int row_start = row_indx[global_row];
		int row_end = row_indx[global_row+1];
		int row_amount = row_end - row_start;
		int row_count = 0;

		if(DEBUG) {
      		printf("Row: %d, Col: %d, row_start: %d, row_end: %d, row_amount: %d, row_count: %d, WPT: %d      From device\n", 
      				global_row, global_col, row_start, row_end, row_amount, row_count, WPT);
      	}

		__shared__ float val_sub[BLOCK_SIZE+PADDING_ROW][BLOCK_SIZE+PADDING_COL];
		__shared__ int col_indx_sub[BLOCK_SIZE+PADDING_ROW][BLOCK_SIZE+PADDING_COL];

		// Initialise the accumulation registers	
		float product_val[WPT];
		for (int w=0; w<WPT; w++) {
        	product_val[w] = 0;
    	}

    	__syncthreads();

    	for(row_count = 0; row_count < row_amount; row_count += TS) {

    		//Need all threads to reach barrier in conditional code
    		for (int w=0; w<WPT; w++) {
    			if((row_count + threadIdx.x + w*RTS) >= row_amount) {
    				val_sub[threadIdx.x + w*RTS][threadIdx.y] = 0;
    				col_indx_sub[threadIdx.x + w*RTS][threadIdx.y] = 0;
    			}
    			else {
    				val_sub[threadIdx.x + w*RTS][threadIdx.y] = val[row_start + row_count + threadIdx.x + w*RTS];
    				col_indx_sub[threadIdx.x + w*RTS][threadIdx.y] = col_indx[row_start + row_count + threadIdx.x + w*RTS];
    			}
    		}
    		__syncthreads();

      		if(DEBUG) {
      			printf("row: %d, col: %d, vals_subs:%f, %f, %f, %f, col_indx_subs: %d, %d, %d, %d \n", global_row, global_col, 
      					val_sub[0][threadIdx.y], val_sub[1][threadIdx.y], val_sub[2][threadIdx.y], val_sub[3][threadIdx.y],
      					col_indx_sub[0][threadIdx.y], col_indx_sub[1][threadIdx.y], 
      					col_indx_sub[2][threadIdx.y], col_indx_sub[3][threadIdx.y]);
      		}

      		for(int k = 0; k < min(BLOCK_SIZE, row_amount - row_count); k++) {
      		//for(int k = 0; k < BLOCK_SIZE; k++) {
    			for (int w=0; w<WPT; w++) {
        			float temp = product_val[w];
        			product_val[w] += val_sub[k][threadIdx.y] * dense[col_indx_sub[k][threadIdx.y]*width2 + global_col + w*RTS];
        			//product_val += val[k] * dense[col_indx[k]*width2 + col];
        			if(DEBUG) {
      					printf("Global_row: %d, Global_col: %d, product_val[w]: %f, previous: temp: %f, k: %d, val_sub[k]: %f, dense[...]: %f, w: %d\n", 
      					global_row, global_col, product_val[w], temp, k, val_sub[k][threadIdx.y], 
      					dense[col_indx_sub[k][threadIdx.y]*width2 + global_col + w], w);
      				}		
				}
    		}

    		__syncthreads();
    	}

    	for (int w=0; w<WPT; w++) {
        	out[global_row*width2+global_col+w*RTS] = product_val[w];
        	//out[global_row*width2+global_col] = product_val;
    	}

	}
}

// =================================================================================================


// Increased the amount of work-per-thread by a factor WPT
__global__ void myGEMM3(const int M, const int N, const int K,
                      	const float* A,
                      	const float* B,
                      	float* C) {
    
    // Thread identifiers
    const int row = threadIdx.x; // Local row ID (max: TS)
    const int col = threadIdx.y; // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*blockIdx.x + row; // Row ID of C (0..M)
    const int globalCol = TS*blockIdx.y + col; // Col ID of C (0..N)

	if(DEBUG) {
      	printf("globalRow: %d, gloabalCol: %d, row: %d, col: %d, WPT: %d\n", 
      			globalRow, globalCol, row, col, WPT);
    }

    // Local memory to fit a tile of TS*TS elements of A and B
    __shared__ float Asub[TS][TS];
    __shared__ float Bsub[TS][TS];

    // Initialise the accumulation registers
    float acc[WPT];
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int w=0; w<WPT; w++) {
            const int tiledRow = TS*t + row;
            const int tiledCol = TS*t + col;
            Asub[col + w*RTS][row] = A[(tiledCol + w*RTS)*M + globalRow];
            Bsub[col + w*RTS][row] = B[(globalCol + w*RTS)*K + tiledRow];
        }

        if(globalRow == 0 && globalCol == 0 && DEBUG) {
      			printf("Global_row: %d, Global_col: %d, row: %d, col: %d, A_subs:%f, %f, %f, %f, B_subs: %f, %f, %f, %f \n", 
      			globalRow, globalCol, row, col, Asub[0][row], Asub[1][row], Asub[2][row], Asub[3][row], 
      			Bsub[col][0], Bsub[col][1], Bsub[col][2], Bsub[col][3]);
      		}

        // Synchronise to make sure the tile is loaded
        //barrier(CLK_LOCAL_MEM_FENCE);
        __syncthreads();

        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            for (int w=0; w<WPT; w++) {
            	float temp = acc[w];
                acc[w] += Asub[k][row] * Bsub[col + w*RTS][k];
                if(globalRow == 0 && globalCol == 0 && DEBUG) {
      					printf("Global_row: %d, Global_col: %d, row: %d, col: %d, acc[w]: %f, previous temp: %f, k: %d, w: %d, Asub[k][row]: %f, Bsub[col + w*RTS][k]: %f\n", 
      					globalRow, globalCol, row, col, acc[w], temp, k, w, Asub[k][row], Bsub[col + w*RTS][k]);
      				}
            }
        }

        // Synchronise before loading the next tile
        //barrier(CLK_LOCAL_MEM_FENCE);
        __syncthreads();
    }

    // Store the final results in C
    for (int w=0; w<WPT; w++) {
        C[(globalCol + w*RTS)*M + globalRow] = acc[w];
    }
}

// =================================================================================================

// Use 2D register blocking (further increase in work per thread)
__global__ void myGEMM6(const int M, const int N, const int K,
                      const  float* A,
                      const  float* B,
                      float* C) {

    // Thread identifiers
    const int tidm = threadIdx.x; // Local row ID (max: TSM/WPTM == RTSM)
    const int tidn = threadIdx.y; // Local col ID (max: TSN/WPTN == RTSN)
    const int offsetM = TSM*blockIdx.x; // Work-group offset
    const int offsetN = TSN*blockIdx.y; // Work-group offset

    // Local memory to fit a tile of A and B
    __shared__ float Asub[TSK][TSM];
    __shared__ float Bsub[TSN][TSK+2];

    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    // Initialise the accumulation registers
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }
    
    // Loop over all tiles
    const int numTiles = K/TSK;
    int t=0;
    do {

        // Load one tile of A and B into local memory
        #pragma unroll
        for (int la=0; la<LPTA; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            //int row = MOD2(id,TSM);
            //int col = DIV2(id,TSM);
            int row = id % TSM;
            int col = id / TSM;
            int tiledIndex = TSK*t + col;
            Asub[col][row] = A[tiledIndex*M + offsetM + row];
            Bsub[row][col] = B[tiledIndex*N + offsetN + row];
        }

        // Synchronise to make sure the tile is loaded
        //barrier(CLK_LOCAL_MEM_FENCE);
        __syncthreads();

        // Loop over the values of a single tile
        for (int k=0; k<TSK; k++) {

            // Cache the values of Bsub in registers
            #pragma unroll
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[col][k];
            }

            // Perform the computation
            #pragma unroll
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[k][row];
                #pragma unroll
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Synchronise before loading the next tile
        //barrier(CLK_LOCAL_MEM_FENCE);
        __syncthreads();

        // Next tile
        t++;
    } while (t<numTiles);

    // Store the final results in C
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}

// =================================================================================================

__global__ void spmm_kernel(const int *rowptr_data, const int *col_data,
                            const float *value_data,
                            const float *mat_data, float *out_data,
                          	int B, int M, int N, int K) {
                          	// Removed arg_out_data argument

  // We ignore blockIdx.y here, because threads
  // across `blockIdx.y` are treated equally.
  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  int row = thread_idx >> 5;            // thread_idx / 32
  int lane_idx = thread_idx & (32 - 1); // thread_idx % 32
  int batch_idx = row / M;

  // Compute the column index of `mat` in which the thread is operating.
  int mat_col_idx = lane_idx + (blockIdx.y << 5);

  // Compute the output index (row-major order).
  int out_idx = row * K + mat_col_idx;
  if(out_idx < 100 && DEBUG_extra) {
      	printf("out_idx: %d\n", out_idx);
  }

  // Helper arrays for warp communication.
  int mat_row, mat_rows[32];
  float val, vals[HAS_VALUE ? 32 : 1];

  // Do not aggregate/write across the Y-axis (lane_idx < leftover).
  int leftover = K - (blockIdx.y << 5);

  if (batch_idx < B) {
    int row_start = __ldg(rowptr_data + (row % M));
    int row_end = __ldg(rowptr_data + (row % M) + 1);
    int col_idx = row_start + lane_idx;

    // scalar_t result = Reducer<scalar_t, REDUCE>::init();
    float result = 0;
    //int arg;

    // Iterate over all `col` indices in parallel within a warp.
    for (int c = row_start; c < row_end; c += 32) {

      if (col_idx < row_end) {
        // Coalesced memory access into `col` and `val`.
        mat_row = __ldg(col_data + col_idx) * K;
        if (HAS_VALUE)
          val = __ldg(value_data + col_idx);
      } else {
        mat_row = -1;
        if (HAS_VALUE)
          val = 0;
      }
      col_idx += 32;

#pragma unroll
      for (int i = 0; i < 32; i++) {
        // Communication between all threads in a warp.
        mat_rows[i] = __shfl_sync(FULL_MASK, mat_row, i);
        if (HAS_VALUE)
          vals[i] = __shfl_sync(FULL_MASK, val, i);
      }

#pragma unroll
      for (int i = 0; i < 32; i++) {
        if (lane_idx < leftover && mat_rows[i] != -1) {
          // Coalesced memory access into `mat`.
          val = __ldg(mat_data + batch_idx * N * K + mat_rows[i] + mat_col_idx);
          if (HAS_VALUE)
            val = vals[i] * val;
          // Reducer<scalar_t, REDUCE>::update(&result, val, &arg, c + i);
          float temp = result;
          result += val;
          if(out_idx == INSPECT && DEBUG) {
      			printf("Out_idx: %d, Row: %d, Lane_idx: %d, Batch_idx: %d, Mat_col_idx: %d, Thread: %d, Block.x: %d, Block.y: %d, prev result: %07.2f, result: %07.2f, val: %07.2f, vals[i]: %07.2f, i: %d\n", 
      			out_idx, row, lane_idx, batch_idx, mat_col_idx, threadIdx.x, blockIdx.x, blockIdx.y, temp, result, val, 0.0f, i);
      		}
        }
      }
    }

    if (lane_idx < leftover) {
      // Coalesced write into `out`.
      // Reducer<scalar_t, REDUCE>::write(out_data + out_idx, result,
      //                                  arg_out_data + out_idx, arg,
      //                                  row_end - row_start);
      out_data[out_idx] = result;
      if(out_idx == INSPECT && DEBUG) {
      	printf("out_idx: %d, result: %f\n", out_idx, result);
      }
    }
  }
}

// =================================================================================================

__global__ void spmm_kernel_wo_shfl(const int *rowptr_data, const int *col_data,
                            		const float *value_data,
                            		const float *mat_data, float *out_data,
                          			int B, int M, int N, int K) {
                          			// Removed arg_out_data argument

    // shared memory as shfl_sync replacement
  __shared__ int mat_rows[THREADS];
  __shared__ float vals[THREADS];

  int mat_row;
  float val;


  // We ignore blockIdx.y here, because threads
  // across `blockIdx.y` are treated equally.
  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  // Initiate values to 0
  mat_rows[thread_idx] = 0;
  vals[thread_idx] = 0;
  //__syncthreads();

  int row = thread_idx >> 5;            // thread_idx / 32
  int lane_idx = thread_idx & (32 - 1); // thread_idx % 32
  int batch_idx = row/ M;

  // pretending to use 32 size warps by simulating 32 columns and 8 rows
  int localrow = threadIdx.x / 32;
  //int localcol = threadIdx.x % 32;

  // Compute the column index of `mat` in which the thread is operating.
  int mat_col_idx = lane_idx + (blockIdx.y << 5);

  // Compute the output index (row-major order).
  int out_idx = row * K + mat_col_idx;
  if(out_idx < 100 && DEBUG_extra && 0) {
  		//__syncthreads();
      	printf("out_idx: %d\n", out_idx);
      	//__syncthreads();
  }

  // Do not aggregate/write across the Y-axis (lane_idx < leftover).
  int leftover = K - (blockIdx.y << 5);

  if (batch_idx < B) {
    int row_start = rowptr_data[row % M];
    int row_end = rowptr_data[(row % M) + 1];
    int col_idx = row_start + lane_idx;

    // scalar_t result = Reducer<scalar_t, REDUCE>::init();
    float result = 0;
    float val_temp = 0;
    //int arg;

    // Iterate over all `col` indices in parallel within a warp.
    for (int c = row_start; c < row_end; c += 32) {

      if (col_idx < row_end) {
        // Coalesced memory access into `col` and `val`.
        mat_row = col_data[col_idx] * K;
        if (HAS_VALUE)
          val = value_data[col_idx];
      } else {
        mat_row = -1;
        if (HAS_VALUE)
          val = 0;
      }
      col_idx += 32;

//...
	  mat_rows[threadIdx.x] = mat_row;
  	  vals[threadIdx.x] = val;
  	  //__syncthreads();
  	  if(out_idx == INSPECT && DEBUG) {
  	  	//__syncthreads();
        printf("%d, %d, %d, %d;   %f, %f, %f, %f\n%d, %d, %d, %d;   %f, %f, %f, %f...\n", 
        mat_rows[0], mat_rows[1], mat_rows[2], mat_rows[3], vals[0], vals[1], vals[2], vals[3],
        mat_rows[0 + 32], mat_rows[1 + 32], mat_rows[2 + 32], mat_rows[3 + 32], vals[0 + 32], vals[1 + 32], vals[2 + 32], vals[3 + 32]);
        //__syncthreads();
      }

#pragma unroll
      for (int i = 0; i < 32; i++) {
        if(out_idx == INSPECT && DEBUG_extra) {
          //__syncthreads();
          printf("i: %d, out_idx: %d, thread: %d\n", i, out_idx, threadIdx.x);
          //__syncthreads();
        }
        if (lane_idx < leftover && mat_rows[localrow*32 + i] != -1) {
          if(out_idx == INSPECT && DEBUG_extra) {
          	//__syncthreads();
          	printf("i(inside): %d, out_idx: %d, thread: %d\n", i, out_idx, threadIdx.x);
          	//__syncthreads();
          }
          // Coalesced memory access into `mat`.
          val = mat_data[batch_idx * N * K + mat_rows[localrow*32 + i] + mat_col_idx];
          if (HAS_VALUE)
          	val_temp = val;
            val = vals[localrow*32 + i] * val;
          // Reducer<scalar_t, REDUCE>::update(&result, val, &arg, c + i);
          float temp = result;
          result += val;
          if(out_idx == INSPECT && DEBUG) {
            //__syncthreads();
      		printf("Out_idx: %d, Thread: %d, Block.x: %d, Block.y: %d, prev result: %07.2f, result: %07.2f, sparse_val: %07.2f, dense_val: %07.2f, i: %d\n", 
      		out_idx, threadIdx.x, blockIdx.x, blockIdx.y, temp, result, vals[localrow*32 + i], val_temp, i);
      		//__syncthreads();
      	  }
        }
      }
    }

    if (lane_idx < leftover) {
      // Coalesced write into `out`.
      // Reducer<scalar_t, REDUCE>::write(out_data + out_idx, result,
      //                                  arg_out_data + out_idx, arg,
      //                                  row_end - row_start);
      out_data[out_idx] = result;
      if(out_idx == INSPECT && DEBUG) {
        //__syncthreads();
      	printf("out_idx: %d, result: %f\n", out_idx, result);
      	//__syncthreads();
      }
    }
  }
}

// =================================================================================================

// Simple transpose kernel for a P * Q matrix
__global__ void transpose(const int P, const int Q,
                          const float* input,
                          float* output) {
    
    // Thread identifiers
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ID0 = blockIdx.x*TRANSPOSEX + tx; // 0..P
    const int ID1 = blockIdx.y*TRANSPOSEY + ty; // 0..Q

    // Set-up the local memory for shuffling
    __shared__ float buffer[TRANSPOSEX][TRANSPOSEY];

    // Swap the x and y coordinates to perform the rotation (coalesced)
    if (ID0 < P && ID1 < Q) {
        buffer[ty][tx] = input[ID1*P + ID0];
    }

    // Synchronise all threads
    // barrier(CLK_LOCAL_MEM_FENCE);
    __syncthreads();

    // We don't have to swap the x and y thread indices here,
    // because that's already done in the local memory
    const int newID0 = blockIdx.y*TRANSPOSEY + tx;
    const int newID1 = blockIdx.x*TRANSPOSEX + ty;

    // Store the transposed result (coalesced)
    if (newID0 < Q && newID1 < P) {
        output[newID1*Q + newID0] = buffer[tx][ty];
    }
}

// =================================================================================================