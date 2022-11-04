#!/usr/bin/bash

for i in `cat list.txt`
do
CUDA_VISIBLE_DEVICES=1 python exp_runner.py --conf confs/insect_white_bkgd.conf --case $1/wobox &&
python exp_runner.py --conf confs/insect_white_bkgd.conf --case $1/wobox --mode validate_mesh
done