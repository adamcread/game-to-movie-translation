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

if [ $2 = 'mask' ]
then 
    mask=1
elif [ $2 = 'no-mask' ]
then 
    mask=0
fi

if [ $3 = 'sample' ]
then 
    sample=1
elif [ $3 = 'no-sample' ]
then 
    sample=0
fi


python3 ./cut/test.py \
    --dataroot "./dataset/frames/test/"\
    --direction $1 \
    --checkpoints_dir "./cut/checkpoints/" \
    --name ${name}"_"$2"_"$3 \
    --phase "train" \
    --num_test 796 \
    --load_size 1920 \
    --crop_size 1920 \
    --no_flip \
    --mask 0