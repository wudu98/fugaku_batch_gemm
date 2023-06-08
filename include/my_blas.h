#ifndef _MY_BLAS_H_
#define _MY_BLAS_H_

#include <cblas.h>

void my_blas_batch_sgemm(const int batch_count, const int *batch_size, const int *batch_head, const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const int* m, const int* n, const int* k, const float* alpha, const float ** a, const int* lda, const float ** b, const int* ldb, const float* beta, float ** c, const int* ldc);

#endif