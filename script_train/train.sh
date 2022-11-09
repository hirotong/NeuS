#!/usr/bin/bash

CONF=$2
CASE=$3

echo "training on GPU:$1, with config $CONF on $CASE"
CUDA_VISIBLE_DEVICES=$1 python exp_runner.py --conf $CONF --case $CASE &&
python exp_runner.py --conf $CONF --case $CASE --mode validate_mesh -r 512
done