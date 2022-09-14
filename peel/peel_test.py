import numpy as np
import cv2
from scipy import stats
from psd_tools import PSDImage
from mmseg.apis import inference_segmentor, init_segmentor
from layer_tree import LayerTree
from image_creator import ImageCreator


def update_mask_list(mask_list, descendant_nodes, model, rgb_img):

    result = inference_segmentor(model, rgb_img)[0]

    # 各画素の対応レイヤを記録した画像を生成
    layer_idx_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype='uint16')
    for i, node in enumerate(descendant_nodes):
        if node.layer_data.visible:
            _, _, _, alpha = node.layer_data.img.split()
            alpha = np.array(alpha)
            layer_idx_img[node.layer_data.bbox[1]: node.layer_data.bbox[3],
            node.layer_data.bbox[0]: node.layer_data.bbox[2]][alpha >= 100] = i
    layer_indices = np.unique(layer_idx_img)

    # マスクリストに推論結果を記録
    for layer_idx in layer_indices:
        node = descendant_nodes[layer_idx]
        bbox = node.layer_data.bbox
        clipped_layer_idx_img = layer_idx_img[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        clipped_result = result[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        mask_list[layer_idx][clipped_layer_idx_img == layer_idx] = clipped_result[clipped_layer_idx_img == layer_idx]


def main():

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

    # 推論
    image_creator = ImageCreator()
    rgb_img = image_creator.create_image(layer_tree, label_flag=False)
    rgb_img = np.array(rgb_img)[:, :, :3][:, :, ::-1]

    update_mask_list(mask_list, descendant_nodes, model, rgb_img)

    # 各レイヤの最終的な推論ラベルを決定
    infer_labels = []
    for mask in mask_list:
        infer_label, _ = stats.mode(mask[mask > 0])
        if infer_label.size > 0:
            infer_labels.append(int(infer_label))
        else:
            infer_labels.append(0)

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
