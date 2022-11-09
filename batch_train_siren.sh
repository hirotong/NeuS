#!/bin/bash -l
#SBATCH --job-name=neus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40gb
#SBATCH --time=40:00:00 
#SBATCH --gres=gpu:1    # Number of GPUs (per node)

##SBATCH --mail-type=ALL
##SBATCH --mail-user=jinguang.tong@csiro.au
 
#This is a comment
#Anything above this line starting with #SBATCH is a set of instructions for the batch system e.g. how much memory you want.
#Do NOT modify "nodes=1" unless you know what you're doing - most "desktop" programs can't use more than one node
 
#The below is what you want to actually run/do

echo "Dataset $1"

echo "number of cores is $SLURM_NTASKS"
echo "job name is $SLURM_JOB_NAME"
module load miniconda3
conda activate pytorch3d
python exp_runner.py --conf confs/insect_use_white_bkgd_wmask_siren.conf --case $1 &&
python exp_runner.py --conf confs/insect_use_white_bkgd_wmask_siren.conf --case $1 --mode validate_mesh -r 512
sleep 120
