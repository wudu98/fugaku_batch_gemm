CC=fcc
CCFLAG=-std=gnu11 -Nclang -Kfast,ocl,openmp -I../include -L../ori_blas_src
BLAS=-SSL2BLAMP -lm -lfjomp -loriblas
SRC=batch_gemm_benchmark.c
EXE=batch_gemm_benchmark

all:
	$(CC) $(SRC) $(CCFLAG) $(BLAS) -o $(EXE)