CC=fcc
CCFLAG=-std=gnu11 -Nclang -Kfast,ocl,openmp -fPIC
BLAS=-SSL2BLAMP -lm -lfjomp

AR=ar

LIB_BATCHED_BLAS = libmyblas.a
LIB_BATCHED_BLAS_SO = libmyblas.so

OBJS_SRC =  \
my_blas.o \

OBJS = $(OBJS_SRC)

all : $(LIB_BATCHED_BLAS) $(LIB_BATCHED_BLAS_SO)
$(LIB_BATCHED_BLAS) : $(OBJS)
	$(AR) r $(LIB_BATCHED_BLAS) $(OBJS)
$(LIB_BATCHED_BLAS_SO) : $(OBJS)
	$(CC) -fPIC -shared -o $(LIB_BATCHED_BLAS_SO) $(OBJS)

.c.o :
	$(CC) $(CCFLAG) -c $< $(DEF)

clean :
	rm -f $(LIB_BATCHED_BLAS) $(LIB_BATCHED_BLAS_SO) *.o