// Compile: fccpx -Nclang -Kopenmp gemm-gench.c -SSL2BLAMP
// Benchmark for (NxK) x (KxM) -> (NxM) matmul in col-major

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

// typedef double real;
typedef float real;
// typedef __fp16 real;

void dgemm_();
void sgemm_();
void fjblas_gemm_r16_();

void matmul(int m, int n, int k, real *a, int lda, real *b, int ldb, real *c, int ldc){
	real one  = 1.0;
	real zero = 0.0;
	
	if(8 == sizeof(real)) dgemm_          ("N", "N", &m, &n, &k, &one, a, &lda, b, &ldb, &zero, c, &ldc);
	if(4 == sizeof(real)) sgemm_          ("N", "N", &m, &n, &k, &one, a, &lda, b, &ldb, &zero, c, &ldc);
	if(2 == sizeof(real)) fjblas_gemm_r16_("N", "N", &m, &n, &k, &one, a, &lda, b, &ldb, &zero, c, &ldc);
}

double fp_peak(){
	int vlen = 64 / sizeof(real);
	int flop = vlen * 4; // dual fma

	int ncore;
#pragma omp parallel
#pragma omp master
	ncore = omp_get_num_threads();

	double Gflops = 2.0 * ncore * flop;

	printf("%d threads, peak: %f GFLOPs\n", ncore, Gflops);
	return Gflops;
}

int main(int ac, char **av){
	int M, N, K, mpad=96, kpad=0, iter=10;
	int do_verify=1;

	if(ac>1) M    = atoi(av[1]);
	if(ac>2) N    = atoi(av[2]);
	if(ac>3) K    = atoi(av[3]);
	if(ac>4) mpad = atoi(av[4]);
	if(ac>5) kpad = atoi(av[5]);
	if(ac>6) iter = atoi(av[6]);
	if(ac>7) do_verify = atoi(av[7]);
	assert(ac>3);

	const int lda =  M + mpad;
	const int ldc =  M + mpad;
	const int ldb =  K + kpad;

	const size_t a_alloc = sizeof(real) * lda * K;
	const size_t b_alloc = sizeof(real) * ldb * N;
	const size_t c_alloc = sizeof(real) * ldc * N;

	printf("A (alloc) : %d MiB\n", (int)(a_alloc >> 20));
	printf("B (alloc) : %d MiB\n", (int)(b_alloc >> 20));
	printf("C (alloc) : %d MiB\n", (int)(c_alloc >> 20));

	double peak = fp_peak();

	size_t align = 256;
	real *a    = aligned_alloc(align, a_alloc); assert(a);
	real *b    = aligned_alloc(align, b_alloc); assert(b);
	real *c    = aligned_alloc(align, c_alloc); assert(c);
	real *cref = aligned_alloc(align, c_alloc); assert(cref);

	// initialize
#define A(i,j) a   [i + lda*j]
#define B(i,j) b   [i + ldb*j]
#define C(i,j) cref[i + ldc*j]
	srand48(20230512);
	for(int i=0; i<M; i++) for(int k=0; k<K; k++) A(i,k) = drand48() - 0.5;
	for(int k=0; k<K; k++) for(int j=0; j<N; j++) B(k,j) = drand48() - 0.5;

	// dry run
	matmul(M, N, K, a, lda, b, ldb, c, ldc);

	// reference matmul
	if(do_verify){
#pragma omp parallel for
		for(int i=0; i<M; i++){
			for(int j=0; j<N; j++){
				real sum = 0.0;
				for(int k=0; k<K; k++){
					sum += A(i,k) * B(k,j);
				}
				C(i,j) = sum;
			}
		}


		double err_max = 0.0;
		for(int i=0; i<M; i++){
			for(int j=0; j<N; j++){
				double val = c   [i + ldc*j];
				double ref = cref[i + ldc*j];
				double err = fabs(val - ref);
				err_max = fmax(err_max, err);
			}
		}
		printf("verify: err_max=%e\n", err_max);
	}

#if 0
	for(int i=0; i<10; i++) for(int j=0; j<10; j++){
		double val = c   [i + ldc*j];
		double ref = cref[i + ldc*j];
		printf("(%d,%d): %e %e\n", i, j, ref, val);
	}
#endif

	// benchmark
	double dt[iter];
	for(int n=0; n<iter; n++){
		double t0 = omp_get_wtime();
		matmul(M, N, K, a, lda, b, ldb, c, ldc);
		double t1 = omp_get_wtime();

		dt[n] = t1 - t0;
	}
	for(int n=0; n<iter; n++){
		double Gflops = 2.0 * M * N * K / dt[n] * 1.e-9; 
		double ratio = Gflops / peak;
		printf("%e sec, %f Gflops, eff=%f%%\n", dt[n], Gflops, 100.*ratio);
	}

	return 0;
}