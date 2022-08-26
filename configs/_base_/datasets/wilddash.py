# dataset settings
dataset_type = 'StandardDataset'
data_root = 'data/custom_datasets'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)  # h,w
img_scale = (1920, 1080)  # w,h
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img/train/wilddash/',
        ann_dir='ann/train/wilddash/',
        classes=['road', 'sidewalk', 'vegetation', 'terrain'],
        palette=[[128, 64, 128], [244, 35, 232], [107, 142, 35], [152, 251, 152]],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img/val/wilddash/',
        ann_dir='ann/val/wilddash/',
        classes=['road', 'sidewalk', 'vegetation', 'terrain'],
        palette=[[128, 64, 128], [244, 35, 232], [107, 142, 35], [152, 251, 152]],
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img/val/wilddash/',
        ann_dir='ann/val/wilddash/',
        classes=['road', 'sidewalk', 'vegetation', 'terrain'],
        palette=[[128, 64, 128], [244, 35, 232], [107, 142, 35], [152, 251, 152]],
        pipeline=test_pipeline))

custom_imports = dict(imports=['mmseg.datasets.standard_dataset'],
                      allow_failed_imports=False)
