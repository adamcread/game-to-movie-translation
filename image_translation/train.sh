#!/bin/bash

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
    load_size=600
    crop_size=512
elif [ $3 = 'no-sample' ]
then 
    sample=0
    load_size=512
    crop_size=512
fi


python3 ./cut/train.py \
    --dataroot "./dataset/frames/train/"\
    --name ${name}"_"$2"_"$3 \
    --batch_size 1 \
    --direction $1 \
    --CUT_mode CUT \
    --phase "train" \
    --mask ${mask} \
    --sample_file ${sample} \
    --load_size ${load_size} \
    --crop_size ${crop_size} \
    --n_epochs $4 \
    --display_id 0 \
    --continue_train 
