#!/bin/bash
set -e

module switch lang/tcsds-1.2.36

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp; pwd`
cd ${PROJECT_ROOT}

threads=$1
export OMP_NUM_THREADS=${threads}

MPIEXEC=""
if [ $threads -eq 48 ]; then
	MPIEXEC="mpiexec -mca plm_ple_memory_allocation_policy interleave_all"
fi

PS=128
HS=7168
SL=2048
NH=56
TB=1
PD=(1 2 4 8 16 32)

cd ./benchmark
make -s

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.

for (( i=0; i<6; i++))
do
    B=1
    M=${SL}
    N=$(( ${PS} * ${NH} * 3 / ${PD[$i]} ))
    K=${HS}
    echo -n $TB "," $B "," $M "," $N "," $K ","
    $MPIEXEC ./batch_gemm_benchmark $TB $B $M $N $K
done

for (( i=0; i<4; i++))
do
    B=$(( ${NH} / ${PD[$i]} ))
    M=${SL}
    N=${SL}
    K=${PS}
    echo -n $TB "," $B "," $M "," $N "," $K ","
    $MPIEXEC ./batch_gemm_benchmark $TB $B $M $N $K
done

for (( i=0; i<4; i++))
do
    B=$(( ${NH} / ${PD[$i]} ))
    M=${SL}
    N=${PS}
    K=${SL}
    echo -n $TB "," $B "," $M "," $N "," $K ","
    $MPIEXEC ./batch_gemm_benchmark $TB $B $M $N $K
done

for (( i=0; i<6; i++))
do
    B=1
    M=${SL}
    N=${HS}
    K=$(( ${HS} / ${PD[$i]} ))
    echo -n $TB "," $B "," $M "," $N "," $K ","
    $MPIEXEC ./batch_gemm_benchmark $TB $B $M $N $K
done

for (( i=0; i<6; i++))
do
    B=1
    M=${SL}
    N=$(( 4 * ${HS} / ${PD[$i]} ))
    K=${HS}
    echo -n $TB "," $B "," $M "," $N "," $K ","
    $MPIEXEC ./batch_gemm_benchmark $TB $B $M $N $K
done

for (( i=0; i<6; i++))
do
    B=1
    M=${SL}
    N=${HS}
    K=$(( 4 * ${HS} / ${PD[$i]} ))
    echo -n $TB "," $B "," $M "," $N "," $K ","
    $MPIEXEC ./batch_gemm_benchmark $TB $B $M $N $K
done
