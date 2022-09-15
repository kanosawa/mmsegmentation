import os
import numpy as np
import cv2
import torch
from scipy import stats
from psd_tools import PSDImage
from mmseg.apis import inference_segmentor, init_segmentor
from layer_tree import LayerTree
from image_creator import ImageCreator
from node_data_for_peel import NodeDataForPeel
from mmseg.apis.inference import LoadImage, Compose
from mmcv.parallel import collate, scatter
from peel_test import extract_descendant_nodes, make_layer_idx_img, update_infer_mask_list


def main():

    config = '../configs/pspnet/pspnet_r50-d8_480x360_live2d.py'
    checkpoint = '../work_dirs/live2d/latest.pth'
    model = init_segmentor(config, checkpoint, device='cuda:0')

    psd = PSDImage.open('hiyori.psd')
    layer_tree = LayerTree(psd, 1.0)
    descendant_nodes = extract_descendant_nodes(layer_tree)

    # マージ後推論結果のリスト
    infer_mask_list = []
    for node in descendant_nodes:
        size = node.layer_data.img.size
        infer_mask_list.append(np.zeros((size[1], size[0]), dtype='uint8'))

    # RGB画像
    image_creator = ImageCreator()
    rgb_img = image_creator.create_image(layer_tree, label_flag=False)
    rgb_img = np.array(rgb_img)[:, :, :3][:, :, ::-1]

    # 各画素の対応レイヤを記録した画像
    prev_layer_idx_img = make_layer_idx_img(descendant_nodes, (rgb_img.shape[0], rgb_img.shape[1]))

    # 推論とマスクリストの更新
    update_infer_mask_list(infer_mask_list, descendant_nodes, model, rgb_img, prev_layer_idx_img)

    # 各レイヤの最終的な推論ラベルを決定
    infer_labels = []
    for mask in infer_mask_list:
        infer_label, _ = stats.mode(mask[mask > 0])
        if infer_label.size > 0:
            infer_labels.append(int(infer_label))
        else:
            if len(infer_labels) > 0:
                infer_labels.append(infer_labels[-1])

    if infer_labels[0] == 0:
        infer_labels[0] = infer_labels[1]

    print('hoge')


if __name__ == '__main__':
    main()