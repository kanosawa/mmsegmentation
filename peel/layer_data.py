from PIL import Image


class LayerData:

    def __init__(self, layer, scale):

        self.visible = layer.visible
        self.is_group = layer.is_group()
        self.label = 0

        if not self.is_group:
            orig_img = layer.composite()
            size = (round(orig_img.width * scale), round(orig_img.height * scale))
            self.img = orig_img.resize(size, Image.BICUBIC)
            x, y = round(layer.bbox[0] * scale), round(layer.bbox[1] * scale)
            self.bbox = (x, y, x + size[0], y + size[1])
