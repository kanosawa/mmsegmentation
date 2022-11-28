import cv2
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import onnxruntime
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.apis.inference import LoadImage, Compose
from mmcv.parallel import collate, scatter


def test_onnx():

    config = 'configs/fcn/fcn_r50-d8_512x512_160k_live2d.py'
    checkpoint = 'work_dirs/live2d/latest.pth'
    img_name = 'data/live2d/train/hiyori000.png'
    img = Image.open(img_name).convert('RGB')

    session = onnxruntime.InferenceSession('mmseg.onnx', providers=["CUDAExecutionProvider"])
    input_name = session.get_inputs()[0].name

    long_side = 480
    scale = long_side / max(img.width, img.height)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((round(img.height * scale), round(img.width * scale))),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    data = transform(img).unsqueeze(0).numpy()

    logit = session.run(None, {input_name: data})
    tmp1 = logit[0]
    tmp2 = tmp1[0, 1:5]
    result = np.argmax(tmp2, axis=0).astype('uint8')
    result = cv2.resize(result, (img.width, img.height))
    cv2.imwrite('result.png', result.astype('uint8'))



if __name__ == '__main__':
    test_onnx()
