import os
import random
_base_  = '../detectors/detectors_htc_r101_20e_coco.py'
load_from = './checkpoints/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth'

# model settings
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_head=[
            dict(
                type='HTCMaskHead',
                with_conv_res=False,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
        ]
    )
)

movie_root = "../dataset/frames/train/movie/"
movie_references = random.sample([movie_root+x for x in os.listdir(movie_root)], k=500)
game_root = "../dataset/frames/train/game/"
game_references = random.sample([game_root+x for x in os.listdir(game_root)], k=500)
train_transforms = [
    dict(
        type='OneOf',
        transforms= [
            dict(
                type="Compose",
                transforms=[
                    dict(
                        type="CLAHE", always_apply=True),
                    dict(
                        type="FDA",
                        reference_images=game_references,
                        beta_limit=0.01,
                        always_apply=True)
                ]),
            dict(
                type="Compose",
                transforms=[
                    dict(
                        type="GaussNoise", always_apply=True),
                    dict(
                        type="GaussianBlur", always_apply=True),
                    dict(
                        type="HistogramMatching",
                        reference_images=movie_references,
                        always_apply=True)
                ])
        ],
        p=0.5
    )
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Albu', 
        transforms=train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='InstaBoost',
        action_candidate=('normal', 'horizontal', 'skip'),
        action_prob=(1, 0, 0),
        scale=(0.8, 1.2),
        dx=15,
        dy=15,
        theta=(-1, 1),
        color_prob=0.25,
        hflag=False,
        aug_ratio=0.5),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 8),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]

dataset_type = 'COCODataset'
classes = ('person',)
data = dict(
    train=dict(
        img_prefix='../coco/images/train/',
        seg_prefix='../coco/segmentations/train/',
        classes=classes,
        ann_file='../coco/annotations/train_filtered.json',
        pipeline=train_pipeline),
    val=dict(
        img_prefix='../coco/images/val/',
        seg_prefix='../coco/segmentations/val/',
        classes=classes,
        ann_file='../coco/annotations/val_filtered.json')
)

runner = dict(type='EpochBasedRunner', max_epochs=6)
evaluation = dict(metric=['bbox', 'segm'])

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    step=[8, 11])

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))