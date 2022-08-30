_base_ = [
    '../_base_/models/ocrnet_hr18.py', '../_base_/datasets/cityscapes.py',
    '../_base_/customize_runtime.py', '../_base_/schedules/schedule_custom.py'
]


data = dict(
    train=dict(
        classes=['road', 'sidewalk', 'vegetation', 'terrain'],
        palette=[[128, 64, 128], [244, 35, 232], [107, 142, 35], [152, 251, 152]],
        split='img/train/cityscapes/small_train1.txt'),
    val=dict(
        classes=['road', 'sidewalk', 'vegetation', 'terrain'],
        palette=[[128, 64, 128], [244, 35, 232], [107, 142, 35], [152, 251, 152]],
        split='img/val/cityscapes/small_val1.txt'),
    test=dict(
        classes=['road', 'sidewalk', 'vegetation', 'terrain'],
        palette=[[128, 64, 128], [244, 35, 232], [107, 142, 35], [152, 251, 152]],
        split='img/val/cityscapes/small_val1.txt'))
