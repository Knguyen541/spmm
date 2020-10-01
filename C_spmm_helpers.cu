// CC_IO: https://people.sc.fsu.edu/~jburkardt/c_src/cc_io/cc_io.html

void create_random_matrix(int row, int col, float* matrix) 
{
	// https://stackoverflow.com/questions/13589248/generating-random-matrix-in-c

    int i;
    for(i = 0; i<row*col; i++)
    	//https://stackoverflow.com/questions/13408990/how-to-generate-random-float-number-in-c
        matrix[i] = (float)rand()/(float)(RAND_MAX/MAX_VALUE); 
    return;
}

void create_random_sparse_matrix(int row, int col, float sparsity, float* matrix) 
{
	// https://stackoverflow.com/questions/24912764/generating-a-sparse-matrix-in-c
	int t = row*col*sparsity;
	int length = row*col;
	int i;
	for (i = 0; i < t;) {
   		int index = (int) (rand() % length);
   		if (matrix[index] != 0) {  /* something already at this index */
      		continue;         /* skip incrementing and try again */
   		}
  	 	matrix[index] = (float)rand()/(float)(RAND_MAX/MAX_VALUE);
  	 	i++;
	}
}

int count_nnz(int row, int col, float* matrix) 
{
	int nnz = 0; //number of nonzero entries

	int i;
	for(i=0; i<row*col; i++) {
		if (matrix[i] != 0) {
			nnz++;
		}
	}

	return nnz;
}

void dense_to_csr(int nnz, int row, int col, float* matrix, 
				  float* val, int* col_indx, int* row_indx) 
{
	row_indx[0] = 0;
	row_indx[row] = nnz;

	int counter = 0;
	int row_used = 0;

	int i, j;

	for(i=0; i<row; i++) {		
		if(i != 0) {
			row_indx[i] = row_used;
		}
		for(j=0; j<col; j++) {
			if(matrix[i*col + j] != 0) {
				val[counter] = matrix[i*col + j];
				col_indx[counter] = j;
				counter++; 
				row_used++;
			}
		}
	}
}

void print_matrix(int row, int col, float* matrix) 
{
	// https://stackoverflow.com/questions/14166350/how-to-display-a-matrix-in-c
	int a, b;

    for(a=0; a<row; a++) {
        for(b=0; b<col; b++)
            {printf("%07.2f     ", matrix[a*col + b]);}
            printf("\n");
     }
    
    getchar();	
    return;
}

void print_float_array(int length, float* array) 
{
	int a;
	for(a = 0; a < length; a++)
    	printf("%07.2f ", array[a]);
    printf("\n");
}

void print_int_array(int length, int* array) 
{
	int a;
	for(a = 0; a < length; a++)
    	printf("%d ", array[a]);
    printf("\n");
}

// Cheap MatMul to check correctness
int check_mm(float* mat1, float* mat2, float* out_to_check, 
			 int height1, int shared_dim, int width2)
{
	float *result = (float *)malloc(height1 * width2 * sizeof(float));
	int i;
	for(i = 0; i < height1 * width2; i++){
		result[i] = 0; 
	}

	// https://www.programiz.com/c-programming/examples/matrix-multiplication
	// Multiplying first and second matrices and storing it in result
   	for (int i = 0; i < height1; ++i) {
   		for (int j = 0; j < width2; ++j) {
        	for (int k = 0; k < shared_dim; ++k) {
            	result[i*width2 + j] += mat1[i*shared_dim + k] * mat2[k*width2 + j];
        	}
    	}
	} 

	if(DEBUG1) {
		printf("Check Matrix: \n"), print_matrix(height1, width2, result);
	}


	// Good float info for comparison: https://floating-point-gui.de/errors/comparison/
	// But we dont need super fine checking
	for(i = 0; i < height1 * width2; i++){
		if (abs(out_to_check[i] - result[i]) > 0.1){
			printf("I: %d, Out_value: %f, Result_value: %f\n", 
					i, out_to_check[i], result[i]);
			return 0;
		}
	}

	return 1;
}