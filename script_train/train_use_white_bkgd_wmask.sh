#!/usr/bin/bash

for i in `cat list.txt`
do
echo $i
CUDA_VISIBLE_DEVICES=1 python exp_runner.py --conf confs/insect_use_white_bkgd_wmask.conf --case $i/withbox &&
python exp_runner.py --conf confs/insect_white_bkgd.conf --case $i/withbox --mode validate_mesh
done