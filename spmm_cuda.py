from numba import cuda
import numpy as np
from scipy.sparse import random
from scipy import stats

@cuda.jit
def add_kernel(rowptr_data, col_data, out):


def main():
	TS = 32
	M = TS
	N = TS
	K = TS
	THREADS = 256

	sparse = random(N, M, density=0.33, format="csr", random_state=42, dtype = np.float32)
	value = sparse.data
	col = sparse.indices
	rowptr = sparse.indptr

	np.random.seed(42)
	dense = np.random.rand(M,K).astype(np.float32)

	result = sparse @ dense

	localrange = THREADS
	globalrange = (int((32 * B * M + THREADS - 1) / THREADS)*THREADS, int((K + 31) / 32))


if __name__ == "__main__":
	main()