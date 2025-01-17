classes = (
    'Background', 'Face', 'Arm', 'UpperBody', 'LowerBody'
)

palette = [
    [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]
]

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=1000, meta=dict(CLASSES=classes, PALETTE=palette))
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)
