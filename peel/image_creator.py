from PIL import Image
import numpy as np


class ImageCreator:
    def __init__(self, label_color_map=None):
        self.label_color_map = label_color_map

    def create_image(self, layer_tree, label_flag):
        img = Image.new('RGBA', layer_tree.size, (255, 255, 255, 255 if label_flag else 0))
        for node in layer_tree.root_node.children:
            img = self.paste_node_image(img, layer_tree.size, node, label_flag)
        return img

    def paste_node_image(self, img, size, node, label_flag):
        if not node.layer_data.visible:
            return img

        if not node.layer_data.is_group:
            if label_flag:
                img = Image.alpha_composite(img, self.get_node_label_image(node.layer_data, size))
            else:
                img = Image.alpha_composite(img, self.get_node_rgb_image(node.layer_data, size))

        for child_node in node.children:
            img = self.paste_node_image(img, size, child_node, label_flag)

        return img

    def get_node_rgb_image(self, layer_data, size):
        layer_rgb_img = Image.new('RGBA', size, (0, 0, 0, 0))
        layer_rgb_img.paste(layer_data.img, layer_data.bbox)
        return layer_rgb_img

    def get_node_label_image(self, layer_data, size):
        label = layer_data.label
        color = (label, label, label) if self.label_color_map is None else self.label_color_map[label]
        layer_label_img = Image.new('RGB', layer_data.img.size, color)
        _, _, _, alpha = layer_data.img.split()
        alpha = alpha.point(lambda p: p >= 100 and 255)
        layer_label_img.putalpha(alpha)
        whole_label_img = Image.new('RGBA', size, (0, 0, 0, 0))
        whole_label_img.paste(layer_label_img, layer_data.bbox)
        return whole_label_img
