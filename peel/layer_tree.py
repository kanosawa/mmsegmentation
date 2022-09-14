from anytree import Node
from layer_data import LayerData


class LayerTree:
    def __init__(self, psd, scale):
        self.size = (round(psd.width * scale), round(psd.height * scale))
        self.root_node = Node('root', parent=None, layer_data=None)
        for layer in psd:
            self.add_layer(layer, self.root_node, scale)

    def add_layer(self, layer, parent_node, scale):
        current_node = Node(layer.name, parent=parent_node, layer_data=LayerData(layer, scale))
        if layer.is_group():
            for child_layer in layer:
                self.add_layer(child_layer, current_node, scale)

    def input_label_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            idx = 0
            for node in self.root_node.children:
                idx = self.input_label_file_for_node(node, lines, idx)

    def input_label_file_for_node(self, node, lines, idx):
        label = lines[idx].split(',')[1].strip()
        node.layer_data.label = int(label)
        idx += 1
        for child_node in node.children:
            idx = self.input_label_file_for_node(child_node, lines, idx)
        return idx

    def output_label_file(self, filename):
        with open(filename, 'w') as f:
            for node in self.root_node.children:
                self.output_label_file_for_node(node, f)

    def output_label_file_for_node(self, node, f):
        f.write(node.name + ',' + str(node.layer_data.label) + '\n')
        for child_node in node.children:
            self.output_label_file_for_node(child_node, f)

