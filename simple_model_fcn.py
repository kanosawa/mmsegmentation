import numpy as np
import torch
from torch import onnx
import cv2
from torch import nn
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from mmseg.apis import init_segmentor
from PIL import Image
from mmseg.apis.inference import LoadImage, Compose
from mmcv.parallel import collate, scatter


class SimpleModel(nn.Module):
    def __init__(self, orig_model):
        super().__init__()
        self.backbone = orig_model.backbone
        self.decode_head = orig_model.decode_head

    def forward(self, x):
        x = self.backbone(x)
        out = self.decode_head(x)
        return out


def main():

    # config = 'configs/pspnet/pspnet_r50-d8_480x360_live2d.py'
    config = 'configs/fcn/fcn_r50-d8_512x512_160k_live2d.py'
    checkpoint = 'work_dirs/live2d/latest.pth'
    img_name = 'data/live2d/train/chitose000.png'
    opacity = 0.5
    out_file = 'result_.jpg'

    palette = [
        [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]
    ]
    orig_model = init_segmentor(config, checkpoint, device='cuda:0')

    simple_model = SimpleModel(orig_model)

    # img = Image.open(img_name).convert("RGB")
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((256, 256)),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # data = transform(img).unsqueeze(0).cuda()

    cfg = orig_model.cfg
    device = next(orig_model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img_name)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(orig_model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    onnx.export(simple_model, data['img'][0], "mmseg.onnx",
                input_names=["input"],
                dynamic_axes={
                    "input": {0: "batch_size", 2: "height", 3: "width"}
                })

    out = simple_model(data['img'][0])
    out = out[0].cpu().detach().numpy()[1:5]
    pred = np.argmax(out, axis=0).astype('uint8')
    pred = cv2.resize(pred, (1432, 2632))
    cv2.imwrite('pred.png', pred)


if __name__ == '__main__':
    main()

