#!/bin/bash 

INPUT_GRAPH="../../../../data/graphs/mtx/benchmark"
NUM_TEST=100
TEST="cpu_opt"

python3 run.py -t $TEST -g $INPUT_GRAPH -n $NUM_TEST -e 0 -m 100 -a 0.85

