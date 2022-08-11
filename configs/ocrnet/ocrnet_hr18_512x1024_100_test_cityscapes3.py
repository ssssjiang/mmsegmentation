_base_ = [
    '../_base_/models/ocrnet_hr18.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_100.py'
]

data_root = 'data/test_cityscapes3/'
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        data_root=data_root),
    val=dict(
        data_root=data_root),
    test=dict(
        data_root=data_root))
