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
#SBATCH --nodelist=gpu7

# job name
#SBATCH --job-name=translation

# Source the bash profile (required to use the module command)
source /etc/profile
module load cuda/11.0-cudnn8.0
source ../venv/bin/activate

if [ $2 = 'mask' ]
then 
    mask=1
elif [ $2 = 'no-mask' ]
then 
    mask=0
fi

if [ $1 = 'AtoB' ]
then
    python3 train.py \
        --dataroot "../dataset/frames/train/"\
        --name "game2movie_"$2 \
        --batch_size 1 \
        --direction "AtoB" \
        --CUT_mode CUT \
        --phase "train" \
        --mask ${mask} \
        --load_size 512 \
        --crop_size 512 \
        --n_epochs $3 \
        --display_id 0 \
        --continue_train 
elif [ $1 = 'BtoA' ]
then 
    python3 train.py \
        --dataroot "../dataset/frames/train/" \
        --name "movie2game_"$2 \
        --batch_size 1 \
        --direction "BtoA" \
        --CUT_mode CUT \
        --phase "train" \
        --mask ${mask} \
        --load_size 512 \
        --crop_size 512 \
        --n_epochs $3 \
        --display_id 0 \
        --continue_train
else
    python3 train.py \
        --dataroot "../dataset/frames/train/" \
        --name "debug" \
        --batch_size 1 \
        --direction "BtoA" \
        --CUT_mode CUT \
        --phase "train" \
        --display_id 0 \
        --gpu_ids -1 \
        --mask ${mask} \
        --load_size 520 \
        --crop_size 512 \
        --sample_file 1
fi