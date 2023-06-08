CC=fcc
CCFLAG=-std=gnu11 -Nclang -Kfast,ocl,openmp -fPIC -I../include/ -L.
BLAS=-SSL2BLAMP -lm -lfjomp -loriblas

AR=ar

LIB_BATCHED_BLAS = liboriblas.a
LIB_BATCHED_BLAS_SO = liboriblas.so

OBJS_SRC =  \
ori_blas.o \

OBJS = $(OBJS_SRC)

SRC=batch_gemm_benchmark.c
EXE=batch_gemm_benchmark

all : $(LIB_BATCHED_BLAS) $(LIB_BATCHED_BLAS_SO) benchmark
$(LIB_BATCHED_BLAS) : $(OBJS)
	$(AR) r $(LIB_BATCHED_BLAS) $(OBJS)
$(LIB_BATCHED_BLAS_SO) : $(OBJS)
	$(CC) -fPIC -shared -o $(LIB_BATCHED_BLAS_SO) $(OBJS)

.c.o :
	$(CC) $(CCFLAG) -c $< $(DEF)

benchmark :
	$(CC) $(SRC) $(CCFLAG) $(BLAS) -o $(EXE)

clean :
	rm -f $(LIB_BATCHED_BLAS) $(LIB_BATCHED_BLAS_SO) *.o