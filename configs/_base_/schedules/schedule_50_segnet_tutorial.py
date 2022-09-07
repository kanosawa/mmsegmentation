classes = (
    'sky', 'Bulding', 'Pole', 'Road_marking', 'Road', 'Pavement', 'Tree',
    'SingSymbole', 'Fence', 'Car', 'Pedestrian', 'Bicyclist'
)

palette = [
    [128, 128, 128], [128, 0, 0], [192, 192, 128], [255, 69, 0], [128, 64, 128], [60, 40, 222],
    [128, 128, 0], [192, 128, 128], [64, 64, 128], [64, 0, 128], [64, 64, 0], [0, 128, 192]
]

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=50, meta=dict(CLASSES=classes, PALETTE=palette))
evaluation = dict(interval=50, metric='mIoU', pre_eval=True)
