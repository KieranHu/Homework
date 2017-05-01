from __future__ import division, print_function, absolute_import
from layer import *


class Graph(object):
    """The graph or network structure of a neural network.

    Arguments:
        config (list): a list of tuples with each tuple contains the name and
            parameters of a layer.

    Attributes:
        config (list): a list of tuples with each tuple contains the name and
            parameters of a layer.
        layers (list): a list of layers. Each layer is a layer object
            instantiated using a class from the "layer" module.
        names (list): a list of layer names.

    """

    def __init__(self, config):
        self.config = config
        self.layers = []
        self.names = []

        for layer_type, layer_params in config:
            self.__check_layer(layer_type)

            layer = self.__create_layer(layer_type, layer_params)
            if layer.name == layer_type:
                layer_index = 0
                for name in self.names[::-1]:
                    if layer_type in name:
                        layer_index = int(name.split("_")[1]) + 1
                        break
                layer.name = layer_type + "_" + str(layer_index)

            self.names.append(layer.name)
            self.layers.append(layer)

    def __getitem__(self, key):
        if type(key) == int:
            return self.layers[key]

        if key not in self.names:
            raise KeyError(
                "{} is not in the graph".format(key)
            )
        return self.layers[self.names.index(key)]

    def __len__(self):
        return len(self.layers)

    def __repr__(self):
        return "\n".join([str(layer) for layer in self.layers])

    def __check_layer(self, layer_type):
        if layer_type not in globals():
            raise NameError(
                "{} is not an valid layer name!".format(layer_type)
            )

    def __create_layer(self, layer_type, layer_params):
        if layer_params:
            return globals()[layer_type](**layer_params)
        else:
            return globals()[layer_type]()
