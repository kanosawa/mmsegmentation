_base_ = [
    '../_base_/models/pspnet_r50-d8_nosync.py', '../_base_/datasets/live2d.py',
    '../_base_/log_interval10_runtime.py', '../_base_/schedules/schedule_50_live2d.py'
]
