// KERNEL
// 1: Normal MatMul
// 2: First SpMatMul
// 3: Second SpMatMul
// 2X: Xth Kernel of Gemm tutorial
// 999: pytorch_sparse cuda source code
#define KERNEL 999

// #define OPENCL 1 //  This is for both OpenCL or Cuda and the macro decides which version to use to rewrite kernels
#define UseShflSync 0 // Can decide here whether to use certain kernels with or without shfl_sync

#define SIZE   8
#define SHARED SIZE
#define SROW   SIZE
#define SCOL   SHARED
#define DROW   SHARED
#define DCOL   SIZE

#define SPARSITY  0.25    //How sparse should the randomly generated matrix be from 0-1
#define MAX_VALUE 10
#define NUM_RUNS  1

#define DEBUG        1    //Some KERNEL cuda prints
#define DEBUG_extra  1    //even more extra KERNEL cuda prints
#define DEBUG1       1    //Shows matrix values
#define DEBUG2       0    //Shows where code reached

#define INSPECT      32    //value to inspect specific thread
#define CHECK        1    //checks correctness with slow cpu matmul

#define BLOCK_SIZE   32
#define PADDING_ROW  0        
#define PADDING_COL  2  // padding for less bank conflicts

#define TS   BLOCK_SIZE
#define WPT  1                         // The amount of work-per-thread, i.e. the thread-coarsening factor
#define RTS  (TS/WPT)                 // The reduced tile-size in one dimension


// Constants for kernels 26 -- 210
#define TSM 128                      // The tile-size in dimension M
#define TSN 128                      // The tile-size in dimension N
#define TSK 16                       // The tile-size in dimension K
#define WPTM 8                       // The amount of work-per-thread in dimension M
#define WPTN 8                       // The amount of work-per-thread in dimension N
#define RTSM (TSM/WPTM)              // The reduced tile-size in dimension M (== number of threads)
#define RTSN (TSN/WPTN)              // The reduced tile-size in dimension N (== number of threads)
#define LPTA ((TSK*WPTM*WPTN)/(TSN)) // The amount of loads-per-thread for A
#define LPTB ((TSK*WPTM*WPTN)/(TSM)) // The amount of loads-per-thread for B

// Constants for the supporting transpose kernel
#define TRANSPOSEX 16
#define TRANSPOSEY 16

#define THREADS 256
#define FULL_MASK 0xffffffff
#define HAS_VALUE 1

