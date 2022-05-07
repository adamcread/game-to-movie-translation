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
#SBATCH --job-name=segmentation_train

# Source the bash profile (required to use the module command)
source /etc/profile
module load cuda/11.0-cudnn8.0
source ../venv/bin/activate


if [ $2 = "continue" ]
then
    python3 train.py \
        --dataroot ./datasets/game2movie/ \
        --checkpoints_dir "./checkpoints/" \
        --direction $1 \
        --name "game2movie_"$1 \
        --model attention_gan \
        --dataset_mode unaligned \
        --pool_size 50 \
        --no_dropout \
        --norm instance \
        --lambda_A 10 \
        --lambda_B 10 \
        --lambda_identity 0.5 \
        --load_size 286 \
        --crop_size 256 \
        --batch_size 4 \
        --niter 80 \
        --niter_decay 0 \
        --display_freq 100 \
        --print_freq 100 \
        --save_epoch_freq 20 \
        --continue_train
elif [ $2 = "no-continue" ]
then
    python3 train.py \
        --dataroot ./datasets/game2movie/ \
        --checkpoints_dir "./checkpoints/" \
        --direction $1 \
        --name "game2movie_"$1 \
        --model attention_gan \
        --dataset_mode unaligned \
        --pool_size 50 \
        --no_dropout \
        --norm instance \
        --lambda_A 10 \
        --lambda_B 10 \
        --lambda_identity 0.5 \
        --load_size 286 \
        --crop_size 256 \
        --batch_size 4 \
        --niter 10 \
        --niter_decay 0 \
        --display_freq 100 \
        --save_epoch_freq 10 \
        --print_freq 100
fi