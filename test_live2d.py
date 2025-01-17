import cv2
import torch
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.apis.inference import LoadImage, Compose
from mmcv.parallel import collate, scatter


def main():

    # config = 'configs/pspnet/pspnet_r50-d8_480x360_live2d.py'
    config = 'configs/fcn/fcn_r50-d8_512x512_160k_live2d.py'
    checkpoint = 'work_dirs/live2d/latest.pth'
    img = 'data/live2d/train/chitose000.png'
    opacity = 0.5
    out_file = 'result_.jpg'

    palette = [
        [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]
    ]
    model = init_segmentor(config, checkpoint, device='cuda:0')


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

    logit = model.encode_decode(img=data['img'][0], img_metas=data['img_metas'])
    result = torch.argmax(logit[0, 1:5], dim=0)
    result = result.cpu().detach().numpy()
    cv2.imwrite('result.png', result.astype('uint8'))



    result = inference_segmentor(model, img)
    # cv2.imwrite('result.png', result[0].astype('uint8'))

    show_result_pyplot(
        model,
        img,
        result,
        palette,
        opacity=opacity,
        out_file=out_file
    )


if __name__ == '__main__':
    main()
