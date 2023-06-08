#include <stdio.h>
#include <stdlib.h>

#include "my_blas.h"

void my_blas_batch_sgemm(const int batch_count, const int *batch_size, const int *batch_head, const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const int* m, const int* n, const int* k, const float* alpha, const float ** a, const int* lda, const float ** b, const int* ldb, const float* beta, float ** c, const int* ldc)
{
    // const int num_threads = omp_get_max_threads();

    for(int i = 0; i < batch_count; i++){
		for(int j = 0; j < batch_size[i]; j++){
			cblas_sgemm(layout, transa, transb, m[i], n[i], k[i], alpha[i], a[batch_head[i]+j], lda[i], b[batch_head[i]+j], ldb[i], beta[i], c[batch_head[i]+j], ldc[i]);
		}
	}
}
