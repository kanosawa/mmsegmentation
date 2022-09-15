import numpy as np


class NodeDataForPeel:
    def __init__(self, node):
        self.area_threshold = 0.6
        self.visible = node.layer_data.visible
        self.label = node.layer_data.label
        self.bbox = node.layer_data.bbox
        self.mask = self.extract_mask(node)
        self.orig_area = np.count_nonzero(self.mask)
        self.done_area = 0

    def extract_whole_mask(self, size):
        whole_mask = np.zeros([size[1], size[0]], dtype='uint8')
        whole_mask[self.bbox[1]: self.bbox[3], self.bbox[0]: self.bbox[2]] = self.mask
        return whole_mask

    def clip_img_by_bbox(self, img):
        return img[self.bbox[1]: self.bbox[3], self.bbox[0]: self.bbox[2]]

    def add_area(self, new_area):
        self.done_area += new_area
        return self.done_area - new_area < self.orig_area * self.area_threshold <= self.done_area

    def extract_mask(self, node):
        _, _, _, mask = node.layer_data.img.split()
        mask = np.array(mask)
        mask[mask > 100] = 255
        return mask
