import os
from glob import glob
import numpy as np
import cv2
from psd_tools import PSDImage
from layer_tree import LayerTree
from node_data_for_peel import NodeDataForPeel
from image_creator import ImageCreator


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


def peel_one_model(data_name, psd_dir):

    label_dir = 'label'
    train_dir = 'train'
    trainanno_dir = 'trainanno_'

    # 子孫nodeを上からリスト化
    psd = PSDImage.open(os.path.join(psd_dir, data_name + '.psd'))
    layer_tree = LayerTree(psd, 1.0)
    layer_tree.input_label_file(os.path.join(label_dir, data_name + '.txt'))
    reversed_descendant_nodes = extract_descendant_nodes(layer_tree)
    reversed_descendant_nodes.reverse()

    # RGB画像とラベル画像を生成
    image_creator = ImageCreator()
    rgb_img = image_creator.create_image(layer_tree, label_flag=False)
    prev_label_img, _, _, _ = image_creator.create_image(layer_tree, label_flag=True).split()
    prev_label_img = np.array(prev_label_img)
    whole_area = np.count_nonzero(prev_label_img < 255)

    # NodeDataForPeelのリストを生成
    node_datas = []
    for node in reversed_descendant_nodes:
        node_datas.append(NodeDataForPeel(node, whole_area))

    # 画像ファイルを保存
    rgb_img.save(os.path.join(train_dir, data_name + '000.png'))
    cv2.imwrite(os.path.join(trainanno_dir, data_name + '000.png'), prev_label_img)

    # 初回のupdate処理（最初から見えてる画素をNodeDataForPeelのmaskから除去）
    diff_mask = (prev_label_img < 255).astype('uint8') * 255
    update(node_datas, diff_mask, psd.size)

    output_idx = 1
    for i, node in enumerate(reversed_descendant_nodes):

        print(i, node.name)

        node.layer_data.visible = 0
        node_datas[i].visible = 0

        current_label_img, _, _, _ = image_creator.create_image(layer_tree, label_flag=True).split()
        current_label_img = np.array(current_label_img)

        diff_mask = (current_label_img - prev_label_img != 0)
        diff_mask = diff_mask.astype('uint8') * 255

        output_flag = update(node_datas, diff_mask, psd.size)
        if output_flag:
            rgb_img = image_creator.create_image(layer_tree, label_flag=False)
            rgb_img.save(os.path.join(train_dir, data_name + '{0:03d}.png'.format(output_idx)))
            cv2.imwrite(os.path.join(trainanno_dir, data_name + '{0:03d}.png'.format(output_idx)), prev_label_img)
            output_idx += 1
        prev_label_img = current_label_img


def main():
    psd_dir = 'psd'
    psd_paths = glob(os.path.join(psd_dir, '*.psd'))
    data_names = list(map(lambda x: os.path.splitext(os.path.basename(x))[0], psd_paths))
    for data_name in data_names:
        peel_one_model(data_name, psd_dir)


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
