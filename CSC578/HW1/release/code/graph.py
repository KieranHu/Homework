from __future__ import division, print_function, absolute_import
from layer import *


class Graph(object):
    """The graph or network structure of a neural network.

    Arguments:
        config(list): a list of tuples with each tuple contains the name and
            parameters of a layer.

    Attributes:
        config(list): a list of tuples with each tuple contains the name and
            parameters of a layer.
        layers(list): a list of layers. Each layer is a layer object
            instantiated using a class from the "layer" module.

    """

    def __init__(self, config):
        self.config = config
        self.layers = []

        for layer_name, layer_params in config:
            self.__check_layer(layer_name)

            layer = self.__create_layer(layer_name, layer_params)
            self.layers.append(layer)

    def __getitem__(self, key):
        return self.layers[key]

    def __str__(self):
        graph_str = ""
        for layer_name, layer_params in self.config:
            graph_str += "{} {}\n".format(layer_name, layer_params)
        return graph_str

    def __check_layer(self, layer_name):
        if layer_name not in globals():
            raise NameError(
                "{} is not an valid layer name!".format(layer_name)
            )

    def __create_layer(self, layer_name, layer_params):
        if layer_params:
            return globals()[layer_name](**layer_params)
        else:
            return globals()[layer_name]()
