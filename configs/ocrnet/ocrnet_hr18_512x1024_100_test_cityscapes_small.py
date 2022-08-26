_base_ = [
    '../_base_/models/ocrnet_hr18.py', '../_base_/datasets/cityscapes.py',
    '../_base_/customize_runtime.py', '../_base_/schedules/schedule_custom.py'
]


data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        split='img/train/cityscapes/small_train1.txt'),
    val=dict(
        split='img/val/cityscapes/small_val1.txt'),
    test=dict(
        split='img/val/cityscapes/small_val1.txt'))
