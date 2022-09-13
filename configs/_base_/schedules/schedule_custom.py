# customize schedule
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)

optimizer_config = dict(
    grad_clip=dict(max_norm=1, norm_type=2))

# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True, start=100)