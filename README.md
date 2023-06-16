# fugaku_gemm

**Compilation**

```
 # run on Computing node
 bash ./build_my_lib.sh
```

**Benchmark**

```
 # run on Computing node
 LD_PRELOAD=$LD_PRELOAD:/path/to/fugaku_batch_gemm/my_blas_src/libmyblas.so bash ./run_gpt_gemm_simple.sh 48
```