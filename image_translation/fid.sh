#!/bin/bash
# X number of nodes with Y number of cores in each node.
#SBATCH -N 1
#SBATCH -c 4

# partition time limit and resource limit for the job
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small	
#SBATCH --mem=28g
#SBATCH --qos=long-high-prio
#SBATCH -t 07-00:00:00
#SBATCH --nodelist=gpu8

# job name
#SBATCH --job-name=translation

# Source the bash profile (required to use the module command)
source /etc/profile
module load cuda/11.0-cudnn8.0
source ./venv/bin/activate

if [ $1 = 'AtoB' ]
then
    name="game2movie"
elif [ $1 = "BtoA" ]
then
    name="movie2game"
fi 


python3 -m pytorch_fid ./results/${name}"_"$2"_"$3/train_latest/images/real_A/ ./results/${name}"_"$2"_"$3/train_latest/images/fake_B/