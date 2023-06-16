#!/bin/bash
set -e

module switch lang/tcsds-1.2.36

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp; pwd`
cd ${PROJECT_ROOT}

cd ./my_blas_src
make clean
make 

