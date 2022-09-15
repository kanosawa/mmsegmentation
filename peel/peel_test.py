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


def infer(model, img):

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
    result = result.cpu().detach().numpy().astype('uint8') + 1
    result = cv2.resize(result, dsize=(img.shape[1], img.shape[0]))
    return result


def update_mask_list(mask_list, descendant_nodes, model, rgb_img, layer_idx_img):

    result = infer(model, rgb_img)

    # マスクリストに推論結果を記録
    for layer_idx in np.unique(layer_idx_img):
        if layer_idx < 0:
            continue
        node = descendant_nodes[layer_idx]
        bbox = node.layer_data.bbox
        clipped_layer_idx_img = layer_idx_img[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        clipped_result = result[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        mask_list[layer_idx][clipped_layer_idx_img == layer_idx] = clipped_result[clipped_layer_idx_img == layer_idx]


def update(node_datas, diff_mask, size):

    output_flag = False
    for node_data in node_datas:
        if node_data.visible:

            # diff_maskとの共通領域を抽出
            whole_node_mask = node_data.extract_whole_mask(size)
            and_mask = cv2.bitwise_and(whole_node_mask, diff_mask)

            # 共通領域があるなら
            if np.any(and_mask == 255):
                and_mask_in_bbox = node_data.clip_img_by_bbox(and_mask)
                new_area = np.count_nonzero(and_mask_in_bbox)
                output_flag |= node_data.add_area(new_area)
                node_data.mask[and_mask_in_bbox == 255] = 0
                diff_mask[and_mask == 255] = 0

    return output_flag


# レイヤインデックスを画素値とする画像を生成
def make_layer_idx_img(descendant_nodes, size):
    layer_idx_img = np.ones(size, dtype='int32') * -1
    for i, node in enumerate(descendant_nodes):
        if node.layer_data.visible:
            _, _, _, alpha = node.layer_data.img.split()
            bbox = node.layer_data.bbox
            alpha = np.array(alpha)
            layer_idx_img[bbox[1]: bbox[3], bbox[0]: bbox[2]][alpha >= 100] = i
    return layer_idx_img


def main():

    label_max_value = 30000

    config = '../configs/pspnet/pspnet_r50-d8_480x360_live2d.py'
    checkpoint = '../work_dirs/live2d/latest.pth'
    model = init_segmentor(config, checkpoint, device='cuda:0')

    psd = PSDImage.open('hiyori.psd')
    layer_tree = LayerTree(psd, 1.0)

    descendant_nodes = extract_descendant_nodes(layer_tree)

    # マージ後推論結果のリスト
    mask_list = []
    for node in descendant_nodes:
        size = node.layer_data.img.size
        mask_list.append(np.zeros((size[1], size[0]), dtype='uint8'))

    # RGB画像
    image_creator = ImageCreator()
    rgb_img = image_creator.create_image(layer_tree, label_flag=False)
    rgb_img = np.array(rgb_img)[:, :, :3][:, :, ::-1]

    # 各画素の対応レイヤを記録した画像
    prev_layer_idx_img = make_layer_idx_img(descendant_nodes, (rgb_img.shape[0], rgb_img.shape[1]))

    # 推論とマスクリストの更新
    update_mask_list(mask_list, descendant_nodes, model, rgb_img, prev_layer_idx_img)

    # NodeDataForPeelのリストを生成
    whole_area = np.count_nonzero(prev_layer_idx_img < label_max_value)
    node_datas = []
    for node in descendant_nodes[::-1]:
        node_datas.append(NodeDataForPeel(node, whole_area))

    output_idx = 1
    for i, node in enumerate(descendant_nodes[::-1]):

        print(i, node.name)

        node.layer_data.visible = 0
        node_datas[i].visible = 0

        current_layer_idx_img = make_layer_idx_img(descendant_nodes, (rgb_img.shape[0], rgb_img.shape[1]))

        diff_mask = (current_layer_idx_img - prev_layer_idx_img != 0)
        diff_mask = diff_mask.astype('uint8') * 255

        output_flag = update(node_datas, diff_mask, psd.size)
        if output_flag:
            rgb_img = image_creator.create_image(layer_tree, label_flag=False)
            rgb_img = np.array(rgb_img)[:, :, :3][:, :, ::-1]
            update_mask_list(mask_list, descendant_nodes, model, rgb_img, prev_layer_idx_img)
            output_idx += 1
        prev_layer_idx_img = current_layer_idx_img

    # 各レイヤの最終的な推論ラベルを決定
    infer_labels = []
    for mask in mask_list:
        infer_label, _ = stats.mode(mask[mask > 0])
        if infer_label.size > 0:
            infer_labels.append(int(infer_label))
        else:
            if len(infer_labels) > 0:
                infer_labels.append(infer_labels[-1])

    if infer_labels[0] == 0:
        infer_labels[0] = infer_labels[1]

    print('hoge')


def extract_descendant_nodes(layer_tree):
    descendants = []
    for node in layer_tree.root_node.children:
        descendants = extrance_descendant_nodes_reg(descendants, node)
    return descendants


def extrance_descendant_nodes_reg(descendants, node):
    children = node.children
    if len(children) == 0:
        descendants.append(node)
    else:
        for child_node in children:
            descendants = extrance_descendant_nodes_reg(descendants, child_node)
    return descendants


if __name__ == '__main__':
    main()
