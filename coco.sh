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

# job name
#SBATCH --job-name=download

# Source the bash profile (required to use the module command)
source /etc/profile
module load cuda/11.0-cudnn8.0
source /venv/bin/activate

cd coco
mkdir segmentations
cd segmentations

# wget http://images.cocodataset.org/zips/train2017.zip
# wget http://images.cocodataset.org/zips/val2017.zip
# wget http://images.cocodataset.org/zips/test2017.zip
# wget http://images.cocodataset.org/zips/unlabeled2017.zip
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

# unzip train2017.zip
# unzip val2017.zip
# unzip test2017.zip
# unzip unlabeled2017.zip
unzip stuffthingmaps_trainval2017.zip


# rm train2017.zip
# rm val2017.zip
# rm test2017.zip
# rm unlabeled2017.zip 
rm stuffthingmaps_trainval2017.zip

cd ../
mkdir segmentations_filtered
cd segmentations_filtered
mkdir train
mkdir val

python3 ../util/filter_images.py \
    --json '../annotations/train_filtered.json' \
    --root '../segmentations/train/' \
    --dest '../segmentations_filtered/train/' \
    --categories 'person' \
    --file_extension 'png'

python3 ../util/filter_images.py \
    --json '../annotations/val_filtered.json' \
    --root '../segmentations/val/' \
    --dest '../segmentations_filtered/val/' \
    --categories 'person' \
    --file_extension 'png'

# cd ../
# wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/image_info_test2017.zip
# wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

# unzip annotations_trainval2017.zip
# unzip stuff_annotations_trainval2017.zip
# unzip image_info_test2017.zip
# unzip image_info_unlabeled2017.zip

# rm annotations_trainval2017.zip
# rm stuff_annotations_trainval2017.zip
# rm image_info_test2017.zip
# rm image_info_unlabeled2017.zip