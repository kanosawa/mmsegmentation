_base_ = [
    '../_base_/models/fcn_r50-d8_nosync.py', '../_base_/datasets/live2d.py',
    '../_base_/log_interval10_runtime.py', '../_base_/schedules/schedule_50_live2d.py'
]
model = dict(
    decode_head=dict(num_classes=5), auxiliary_head=dict(num_classes=5))
