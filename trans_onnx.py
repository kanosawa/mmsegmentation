from mmseg.apis import init_segmentor
import numpy as np
from torch import onnx
from mmseg.apis.inference import LoadImage, Compose
from mmcv.parallel import collate, scatter


def main():

    config = 'configs/fcn/fcn_r50-d8_512x512_160k_live2d.py'
    checkpoint = 'work_dirs/live2d/latest.pth'
    model = init_segmentor(config, checkpoint, device='cuda:0')

    img = np.zeros([128, 128, 3], dtype="float32")

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    onnx.export(model, data['img'], "mmseg.onnx")


if __name__ == '__main__':
    main()
