model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer', arch='tiny', img_size=224, drop_path_rate=0.2),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=768,
        init_cfg=None,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=5, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=5, prob=0.5)
    ]))
rand_increasing_policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(type='Invert'),
    dict(type='Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
    dict(type='Posterize', magnitude_key='bits', magnitude_range=(4, 0)),
    dict(type='Solarize', magnitude_key='thr', magnitude_range=(256, 0)),
    dict(
        type='SolarizeAdd',
        magnitude_key='magnitude',
        magnitude_range=(0, 110)),
    dict(
        type='ColorTransform',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(type='Contrast', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Brightness', magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(
        type='Sharpness', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='horizontal'),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='vertical'),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='horizontal'),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='vertical')
]
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies=[
            dict(type='AutoContrast'),
            dict(type='Equalize'),
            dict(type='Invert'),
            dict(
                type='Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
            dict(
                type='Posterize', magnitude_key='bits',
                magnitude_range=(4, 0)),
            dict(
                type='Solarize', magnitude_key='thr',
                magnitude_range=(256, 0)),
            dict(
                type='SolarizeAdd',
                magnitude_key='magnitude',
                magnitude_range=(0, 110)),
            dict(
                type='ColorTransform',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Contrast',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Brightness',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Sharpness',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Shear',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.3),
                direction='horizontal'),
            dict(
                type='Shear',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.3),
                direction='vertical'),
            dict(
                type='Translate',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.45),
                direction='horizontal'),
            dict(
                type='Translate',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.45),
                direction='vertical')
        ],
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(256, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type='ImageNet',
        data_prefix='../dataset/patches/classified/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RandomResizedCrop',
                size=224,
                backend='pillow',
                interpolation='bicubic'),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='RandAugment',
                policies=[
                    dict(type='AutoContrast'),
                    dict(type='Equalize'),
                    dict(type='Invert'),
                    dict(
                        type='Rotate',
                        magnitude_key='angle',
                        magnitude_range=(0, 30)),
                    dict(
                        type='Posterize',
                        magnitude_key='bits',
                        magnitude_range=(4, 0)),
                    dict(
                        type='Solarize',
                        magnitude_key='thr',
                        magnitude_range=(256, 0)),
                    dict(
                        type='SolarizeAdd',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 110)),
                    dict(
                        type='ColorTransform',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Contrast',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Brightness',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Sharpness',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Shear',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.3),
                        direction='horizontal'),
                    dict(
                        type='Shear',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.3),
                        direction='vertical'),
                    dict(
                        type='Translate',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.45),
                        direction='horizontal'),
                    dict(
                        type='Translate',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.45),
                        direction='vertical')
                ],
                num_policies=2,
                total_level=10,
                magnitude_level=9,
                magnitude_std=0.5,
                hparams=dict(pad_val=[104, 116, 124],
                             interpolation='bicubic')),
            dict(
                type='RandomErasing',
                erase_prob=0.25,
                mode='rand',
                min_area_ratio=0.02,
                max_area_ratio=0.3333333333333333,
                fill_color=[103.53, 116.28, 123.675],
                fill_std=[57.375, 57.12, 58.395]),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ],
        classes='../dataset/annotation/classes.txt'),
    val=dict(
        type='ImageNet',
        data_prefix='../dataset/patches/classified/val/',
        ann_file='../dataset/annotation/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                size=(256, -1),
                backend='pillow',
                interpolation='bicubic'),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        classes='../dataset/annotation/classes.txt'),
    test=dict(
        type='ImageNet',
        data_prefix='../dataset/patches/classified/val/',
        ann_file='../dataset/annotation/classes/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                size=(256, -1),
                backend='pillow',
                interpolation='bicubic'),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        classes='../dataset/annotation/classes.txt'))
evaluation = dict(interval=5, metric='accuracy')
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys=dict({
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    }))
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.05,
    eps=1e-08,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        })))
optimizer_config = dict(grad_clip=dict(max_norm=5.0))
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=0.01,
    warmup='linear',
    warmup_ratio=0.001,
    warmup_iters=20,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './checkpoints/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/swin_tiny'
gpu_ids = [0]
