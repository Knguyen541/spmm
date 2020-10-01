
// =================================================================================================

#if KERNEL == 1

__kernel void spmm_kernel1(const __global float* d_M, const __global float* d_N, __global float* d_P, 
							             const int width, const int width2, const int out_row) {
	int out_col = width2;

	int row = get_global_id(1);
	int col = get_global_id(0);

	if(DEBUG) {
		printf("Thread [%d,%d,%d,%d,%d] \
        		From device\n", get_global_id(0), get_global_id(1), get_group_id(0),row,col);
	}

	if(row<out_row&& col <out_col) {
   		float product_val = 0;
   		for(int k=0;k<width;k++) {
      		product_val += d_M[row*width+k]*d_N[k*width2+col];
      		if(row == 4 && col == 2 && DEBUG) {
      			printf("Producval from thread [%d,%d,%f,%d,%f,%f]\n",
              get_global_id(0), get_global_id(1), product_val, k, d_M[row*width+k], d_N[k*width2+col]);
      		}
   		}
   		d_P[row*width2+col] = product_val;
	}
}

#endif

// =================================================================================================

#if KERNEL == 2

__kernel void spmm_kernel2(const __global float* val, 
                           const __global int* col_indx, const __global int* row_indx, 
                           const __global float* dense, __global float* out, 
                           const int width, const int width2, const int out_row)
{
  int out_col = width2;

  int row = get_global_id(1);
  int col = get_global_id(0);

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

#endif

// =================================================================================================

#if KERNEL == 3

__kernel void spmm_kernel3(const __global float* val, 
                           const __global int* col_indx, const __global int* row_indx, 
                           const __global float* dense, __global float* out, 
                           const int width, const int width2, const int out_row)
{
  int out_col = width2;

  int global_row = get_global_id(1);
  int global_col = get_group_id(0) * get_local_size(0) * WPT + get_local_id(0);

  if(DEBUG) {
        printf("Row: %d, Col: %d, blockIdx.x: %d, blockIdx.y: %d, blockDim.x: %d, blockDim.y: %d, WPT: %d\n", 
            global_row, global_col, get_group_id(0), get_group_id(1), get_local_size(0), get_local_size(1), WPT);
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

    __local float val_sub[BLOCK_SIZE+PADDING_ROW][BLOCK_SIZE+PADDING_COL];
    __local int col_indx_sub[BLOCK_SIZE+PADDING_ROW][BLOCK_SIZE+PADDING_COL];

    // Initialise the accumulation registers  
    float product_val[WPT];
    for (int w=0; w<WPT; w++) {
          product_val[w] = 0;
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      for(row_count = 0; row_count < row_amount; row_count += TS) {

        //Need all threads to reach barrier in conditional code
        for (int w=0; w<WPT; w++) {
          if((row_count + get_local_id(0) + w*RTS) >= row_amount) {
            val_sub[get_local_id(0) + w*RTS][get_local_id(1)] = 0;
            col_indx_sub[get_local_id(0) + w*RTS][get_local_id(1)] = 0;
          }
          else {
            val_sub[get_local_id(0) + w*RTS][get_local_id(1)] = val[row_start + row_count + get_local_id(0) + w*RTS];
            col_indx_sub[get_local_id(0) + w*RTS][get_local_id(1)] = col_indx[row_start + row_count + get_local_id(0) + w*RTS];
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

          if(DEBUG) {
            printf("row: %d, col: %d, vals_subs:%f, %f, %f, %f, col_indx_subs: %d, %d, %d, %d \n", global_row, global_col, 
                val_sub[0][get_local_id(1)], val_sub[1][get_local_id(1)], val_sub[2][get_local_id(1)], val_sub[3][get_local_id(1)],
                col_indx_sub[0][get_local_id(1)], col_indx_sub[1][get_local_id(1)], 
                col_indx_sub[2][get_local_id(1)], col_indx_sub[3][get_local_id(1)]);
          }

          for(int k = 0; k < min(BLOCK_SIZE, row_amount - row_count); k++) {
          //for(int k = 0; k < BLOCK_SIZE; k++) {
          for (int w=0; w<WPT; w++) {
              float temp = product_val[w];
              product_val[w] += val_sub[k][get_local_id(1)] * dense[col_indx_sub[k][get_local_id(1)]*width2 + global_col + w*RTS];
              //product_val += val[k] * dense[col_indx[k]*width2 + col];
              if(DEBUG) {
                printf("Global_row: %d, Global_col: %d, product_val[w]: %f, previous: temp: %f, k: %d, val_sub[k]: %f, dense[...]: %f, w: %d\n", 
                global_row, global_col, product_val[w], temp, k, val_sub[k][get_local_id(1)], 
                dense[col_indx_sub[k][get_local_id(1)]*width2 + global_col + w], w);
              }   
        }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
      }

      for (int w=0; w<WPT; w++) {
          out[global_row*width2+global_col+w*RTS] = product_val[w];
          //out[global_row*width2+global_col] = product_val;
      }

  }
}

#endif

// =================================================================================================