_base_ = [
    '../swin_transformer/swin-tiny_16xb64_in1k.py'
]

load_from = './swin_transformer/checkpoints/swin_pretrained.pth'


model = dict(
    head=dict(
        num_classes=5)
    ,
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=5, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=5, prob=0.5)])
    )

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type='ImageNet',
        classes='./dataset/annotation/classification/classes.txt',
        data_prefix='./dataset/patches/classified/train/'),
    val=dict(
        type='ImageNet',
        classes='./dataset/annotation/classification/classes.txt',
        data_prefix='./dataset/patches/classified/val/',
        ann_file='./dataset/annotation/classification/val.txt'),
    test=dict(
        type='ImageNet',
        classes='./dataset/annotation/classification/classes.txt',
        data_prefix='./dataset/patches/extracted/',
        ann_file='./dataset/annotation/classification/test.txt'))

evaluation = dict(interval=5, metric='accuracy')
gpu_ids = [0]
seed = 123
device = 'cpu'
work_dir='./'