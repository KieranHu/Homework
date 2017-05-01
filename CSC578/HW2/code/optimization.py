"""All the optimization methods go here.

"""

from __future__ import division, print_function, absolute_import
import numpy as np


class SGD(object):
    """Stochastic gradient descent.

    Attributes:
        learning_rate(float): the learning rate to use.

    """

    def __init__(self, learning_rate):
        self.learning_rate = float(learning_rate)

    def __is_temporal(self, layer):
        temporal_layers = ['layer.RNNCell', 'layer.LSTMCell']
        layer_type = str(type(layer)).split("'")[1]
        return layer_type in temporal_layers

    def optimize(self, graph, gradients):
        """ Perform SGD on the network defined by 'graph'

        Args:
            graph (obj): a 'Graph' object that defines the structure of a
                neural network.
            gradients (dict): the computed gradients of all the layer
                parameters in the graph.

        """
        for layer_name in gradients:
            layer = graph[layer_name]

            layer.W -= self.learning_rate * gradients[layer_name]["dv_W"]
            layer.b -= self.learning_rate * gradients[layer_name]["dv_b"]
            if self.__is_temporal(layer):
                layer.U -= self.learning_rate * gradients[layer_name]["dv_U"]


class Adagrad(object):
    """Adagrad.

    Attributes:
        learning_rate (float): the learning rate to use.
        memory (dict): the sum of the squares of the gradients with respect
            to gradients.

    """

    def __init__(self, learning_rate):
        self.learning_rate = float(learning_rate)
        self.memory = {}

    def __is_temporal(self, layer):
        temporal_layers = ['layer.RNNCell', 'layer.LSTMCell']
        layer_type = str(type(layer)).split("'")[1]
        return layer_type in temporal_layers

    def optimize(self, graph, gradients):
        """ Perform SGD on the network defined by 'graph'

        Args:
            graph (obj): a 'Graph' object that defines the structure of a
                neural network.
            gradients (dict): the computed gradients of all the layer
                parameters in the graph.

        """

        if not self.memory:
            for m in gradients:
                self.memory[m] = {}
                for n in gradients[m]:
                    self.memory[m][n] = np.zeros_like(gradients[m][n])

        for m in gradients:
            for n in gradients[m]:
                self.memory[m][n] += gradients[m][n] * gradients[m][n]

        for layer_name in gradients:
            layer = graph[layer_name]

            layer.W -= self.learning_rate * \
                gradients[layer_name]["dv_W"] / \
                np.sqrt(self.memory[layer_name]["dv_W"] + 1e-8)
            layer.b -= self.learning_rate * \
                gradients[layer_name]["dv_b"] / \
                np.sqrt(self.memory[layer_name]["dv_b"] + 1e-8)
            if self.__is_temporal(layer):
                layer.U -= self.learning_rate * \
                    gradients[layer_name]["dv_U"] / \
                    np.sqrt(self.memory[layer_name]["dv_U"] + 1e-8)
