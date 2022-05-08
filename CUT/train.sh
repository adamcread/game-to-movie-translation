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
#SBATCH --nodelist=gpu12

# job name
#SBATCH --job-name=translation

# Source the bash profile (required to use the module command)
source /etc/profile
module load cuda/11.0-cudnn8.0
source ../venv/bin/activate

if [ $1 = 'AtoB' ]
then
    python3 train.py \
        --dataroot "../dataset/frames/train/"\
        --name "game2movie" \
        --direction "AtoB" \
        --CUT_mode CUT \
        --phase "train" \
        # --continue_train
elif [ $1 = 'BtoA' ]
then 
    python3 train.py \
        --dataroot "../dataset/frames/train/" \
        --name "game2movie" \
        --direction "AtoB" \
        --CUT_mode CUT \
        --phase "train" \
        # --continue_train
fi