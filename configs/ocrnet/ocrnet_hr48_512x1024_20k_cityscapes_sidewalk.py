_base_ = [
    '../_base_/models/ocrnet_hr18.py', '../_base_/datasets/cityscapes.py',
    '../_base_/customize_runtime.py', '../_base_/schedules/schedule_custom.py'
]


norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=5,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=-1,
            num_classes=5,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ])

data = dict(
    train=dict(
        classes=['road', 'sidewalk', 'vegetation', 'terrain'],
        palette=[[128, 64, 128], [244, 35, 232], [107, 142, 35], [152, 251, 152]]),
    val=dict(
        classes=['road', 'sidewalk', 'vegetation', 'terrain'],
        palette=[[128, 64, 128], [244, 35, 232], [107, 142, 35], [152, 251, 152]]),
    test=dict(
        img_dir='img/val/',
        ann_dir='ann/val/',
        classes=['road', 'sidewalk', 'vegetation', 'terrain'],
        palette=[[128, 64, 128], [244, 35, 232], [107, 142, 35], [152, 251, 152]],
        split='img/val/small_val_w.txt'))
