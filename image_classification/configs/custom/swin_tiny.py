_base_ = [
    '../swin_transformer/swin-tiny_16xb64_in1k.py'
]

model = dict(
    head=dict(
        num_classes=5)
    )
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type='ImageNet',
        data_prefix='../dataset/patches/pose_classification/train/'),
    val=dict(
        type='ImageNet',
        data_prefix='../dataset/patches/pose_classification/val/images/',
        ann_file='data/imagenet/meta/val.txt'),
    test=dict(
        type='ImageNet',
        data_prefix='../dataset/patches/pose_classification/val/',
        ann_file='data/imagenet/meta/val.txt'))

evaluation = dict(interval=10, metric='accuracy')
