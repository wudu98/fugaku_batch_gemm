// Compile: fccpx -Nclang -Kopenmp batch_gemm_benchmark.c -SSL2BLAMP

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

#include <cblas.h>

// typedef double typ;
typedef float typ;
// typedef __fp16 typ;

// void dgemm_();
// void sgemm_();
// void fjblas_gemm_r16_();

// void matmul(int m, int n, int k, typ alpha, typ *a, int lda, typ *b, int ldb, typ beta, typ *c, int ldc){
// 	if(8 == sizeof(typ)) dgemm_          ("N", "N", &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
// 	if(4 == sizeof(typ)) sgemm_          ("N", "N", &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
// 	if(2 == sizeof(typ)) fjblas_gemm_r16_("N", "N", &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
// }

double fp_peak(){
	int vlen = 64 / sizeof(typ);
	int flop = vlen * 4; // dual fma

	int ncore;
#pragma omp parallel
#pragma omp master
	ncore = omp_get_num_threads();

	double Gflops = 2.0 * ncore * flop;

	printf("%d threads, peak: %f GFLOPs\n", ncore, Gflops);
	return Gflops;
}

int main(int argc, char *argv[]){
	int TB, B, M, N, K, padmn=96, padk=0, iter=5, do_verify=0;

  	if(argc>1) TB   = atoi(argv[1]);
  	if(argc>2) B    = atoi(argv[2]);
	if(argc>3) M    = atoi(argv[3]);
	if(argc>4) N    = atoi(argv[4]);
	if(argc>5) K    = atoi(argv[5]);
	if(argc>6) padmn = atoi(argv[6]);
	if(argc>7) padk  = atoi(argv[7]);
	assert(argc>5);

	printf("%d %d %d %d %d\n", TB, B, M, N, K);

	size_t align = 256;
	int *batch_size    = aligned_alloc(align, sizeof(int) * TB);
	int *batch_head    = aligned_alloc(align, sizeof(int) * TB);
	int *m      = aligned_alloc(align, sizeof(int) * TB);
	int *n      = aligned_alloc(align, sizeof(int) * TB);
	int *k      = aligned_alloc(align, sizeof(int) * TB);
	int *lda    = aligned_alloc(align, sizeof(int) * TB);
	int *ldb    = aligned_alloc(align, sizeof(int) * TB);
	int *ldc    = aligned_alloc(align, sizeof(int) * TB);
	typ *alpha  = aligned_alloc(align, sizeof(typ) * TB);
	typ *beta   = aligned_alloc(align, sizeof(typ) * TB);

	// set batch size
	int total_batch_size = 0;
	for(int i = 0; i < TB; i++){
		batch_size[i] = B;
		total_batch_size += batch_size[i];
	}
	// set batch head
	batch_head[0] = 0;
	for(int i = 1; i < TB; i++){
		batch_head[i] = batch_head[i - 1] + batch_size[i - 1];
	}

	typ **a    = aligned_alloc(align, sizeof(typ *) * total_batch_size); assert(a);
	typ **b    = aligned_alloc(align, sizeof(typ *) * total_batch_size); assert(b);
	typ **c    = aligned_alloc(align, sizeof(typ *) * total_batch_size); assert(c);
	typ **cref = aligned_alloc(align, sizeof(typ *) * total_batch_size); assert(cref);

	CBLAS_LAYOUT layout = CblasRowMajor;
	CBLAS_TRANSPOSE transa = CblasNoTrans;
	CBLAS_TRANSPOSE transb = CblasNoTrans;
	size_t a_alloc, b_alloc, c_alloc;
	for(int i = 0; i < TB; i++){
		m[i] = M;
		n[i] = N;
		k[i] = K;
		alpha[i] = 1.0;
		beta[i]  = 0.0;
		if ((layout == CblasRowMajor && transa == CblasNoTrans)||
			(layout == CblasColMajor && transa == CblasTrans)){
			lda[i] = K + padk;
			a_alloc = sizeof(typ) * m[i] * lda[i];
		}
		else
		{
			lda[i] = M + padmn;
			a_alloc = sizeof(typ) * lda[i] * k[i];
		}
		if ((layout == CblasRowMajor && transb == CblasNoTrans)||
			(layout == CblasColMajor && transb == CblasTrans)){
			ldb[i] = N + padmn;
			b_alloc = sizeof(typ) * k[i] * ldb[i];
		}
		else{
			ldb[i] = K + padk;
			b_alloc = sizeof(typ) * ldb[i] * n[i];
		}
		if (layout == CblasRowMajor){
			ldc[i] = N + padmn;
			c_alloc = sizeof(typ) * m[i] * ldc[i];
		}
		else{
			ldc[i] = M + padmn;
			c_alloc = sizeof(typ) * ldc[i] * n[i];
		}

		for(int j = 0; j < batch_size[i]; j++){
			a[batch_head[i]+j]    = aligned_alloc(align, sizeof(typ) * a_alloc);
			b[batch_head[i]+j]    = aligned_alloc(align, sizeof(typ) * b_alloc);
			c[batch_head[i]+j]    = aligned_alloc(align, sizeof(typ) * c_alloc);
			cref[batch_head[i]+j] = aligned_alloc(align, sizeof(typ) * c_alloc);
			for(int l = 0; l < a_alloc; l++){
				a[batch_head[i]+j][l] = drand48();
			}
			for(int l = 0; l < b_alloc; l++){
				b[batch_head[i]+j][l] = drand48();
			}
			for(int l = 0; l < c_alloc; l++){
				cref[batch_head[i]+j][l] = c[batch_head[i]+j][l] = drand48();
			}
		}
		if (i == 0)	{
			printf("A (alloc) : %d MiB\n", (int)(a_alloc >> 20));
			printf("B (alloc) : %d MiB\n", (int)(b_alloc >> 20));
			printf("C (alloc) : %d MiB\n", (int)(c_alloc >> 20));
		}
	}

	double peak = fp_peak();

	// dry run
	for(int i = 0; i < TB; i++){
		for(int j = 0; j < batch_size[i]; j++){
			cblas_sgemm(layout, transa, transb, m[i], n[i], k[i], alpha[i], a[batch_head[i]+j], lda[i], b[batch_head[i]+j], ldb[i], beta[i], c[batch_head[i]+j], ldc[i]);
		}
	}

// 	// reference matmul
// 	if(do_verify){
// #pragma omp parallel for
// 		for(int i=0; i<M; i++){
// 			for(int j=0; j<N; j++){
// 				typ sum = 0.0;
// 				for(int p=0; k<K; k++){
// 					sum += Ai,k) * B(k,j);
// 				}
// 				C(i,j) = sum;
// 			}
// 		}


// 		double err_max = 0.0;
// 		for(int i=0; i<M; i++){
// 			for(int j=0; j<N; j++){
// 				double val = c   [i + ldc*j];
// 				double ref = cref[i + ldc*j];
// 				double err = fabs(val - ref);
// 				err_max = fmax(err_max, err);
// 			}
// 		}
// 		printf("verify: err_max=%e\n", err_max);
// 	}

	// benchmark
	double dt[iter];
	for(int it=0; it<iter; it++){
		double t0 = omp_get_wtime();
#pragma omp parallel for
		for(int i = 0; i < TB; i++){
			for(int j = 0; j < batch_size[i]; j++){
				cblas_sgemm(layout, transa, transb, m[i], n[i], k[i], alpha[i], a[batch_head[i]+j], lda[i], b[batch_head[i]+j], ldb[i], beta[i], c[batch_head[i]+j], ldc[i]);
			}
		}
		double t1 = omp_get_wtime();

		dt[it] = t1 - t0;
	}
	for(int it=0; it<iter; it++){
		double Gflops = 2.0 * TB * B * M * N * K / dt[it] * 1.e-9; 
		double ratio = Gflops / peak;
		printf("%e sec, %f Gflops, eff=%f%%\n", dt[it], Gflops, 100.*ratio);
	}

	for(int i = 0; i < TB; i++){
    	for(int j = 0; j < batch_size[i]; j++){
			free(a[batch_head[i]+j]);
			free(b[batch_head[i]+j]);
			free(c[batch_head[i]+j]);
			free(cref[batch_head[i]+j]);
		}
	}
	free(a);
	free(b);
	free(c);
	free(cref);
  
	free(batch_size);
	free(batch_head);
	free(m);
	free(n);
	free(k);
	free(lda);
	free(ldb);
	free(ldc);
	free(alpha);
	free(beta);

	return 0;
}