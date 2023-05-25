CC=fcc
CCFLAG=-std=gnu11 -Nclang -Kfast,ocl,openmp
BLAS=-SSL2BLAMP -lm -lfjomp
SRC=batch_gemm_benchmark.c
EXE=batch_gemm_benchmark

all:
	$(CC) $(SRC) $(CCFLAG) $(BLAS) -o $(EXE)