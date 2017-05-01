"""All the networks go there.

"""

from __future__ import division, print_function, absolute_import
import numpy as np
import random


class RNN(object):
    """Recurrent Neural Network.

    Attributes:
        graph(obj): A "Graph" object that describes the layers of the network.

    """

    def __init__(self, graph):
        self.graph = graph

    def __is_temporal(self, layer):
        """Check if a layer is a temporal layer.

        This function check if the layer is a vanilla RNN cell or LSTM cell (
        not implemented). You may use this function to decide the proper
        function signature of a layer's forward or backward methods.

        Args:
            layer (obj): an object instantiated using any layer class in the
                "layer" module.

        Returns:
            bool: The return value. True for temporal layers.

        """

        temporal_layers = ['layer.RNNCell', 'layer.LSTMCell']
        layer_type = str(type(layer)).split("'")[1]
        return layer_type in temporal_layers

    def __has_parameters(self, layer):
        """Check if the layer has parameters.

        You may use this function to decide the proper function signature of
        a layer's forward or backward methods. If worked with the
         "__is_temporal()" function, it can be used to distinguish between a
        temporal layer and a layer with only weights and biases parameters.

        Args:
            layer (obj): an object instantiated using any layer class in the
                "layer" module.

        Returns:
            bool: The return value. True for layers with parameters.

        """

        return hasattr(layer, "W")

    def __initialize_memories(self):
        """Initialize all the hidden states in the network with zeros.

        Returns:
            hs (dict): the initialized hidden states. It is a dict with each
                "key" and "value" pair denotes the layer name and the
                corresponding hidden state, respectively.
        """
        hs = {}
        for layer in self.graph:
            if self.__is_temporal(layer):
                hs[layer.name] = np.zeros((layer.shape[0], 1))
        return hs

    def __initilize_gradients(self):
        """Initialize all the gradients of all the parameters in a network with
        zeros.

        Returns:
            gradients (dict): the initialized gradients. It is a dict with each
                "key" and "value" pair denotes the layer name and the
                corresponding gradients, respectively.

        """

        gradients = {}
        for layer in self.graph:
            if self.__is_temporal(layer):
                gradients[layer.name] = {
                    "dv_W": np.zeros_like(layer.W),
                    "dv_U": np.zeros_like(layer.U),
                    "dv_b": np.zeros_like(layer.b)
                }
            elif self.__has_parameters(layer):
                gradients[layer.name] = {
                    "dv_W": np.zeros_like(layer.W),
                    "dv_b": np.zeros_like(layer.b)
                }
        return gradients

    def backpropagate(self, loss, outputs, gts):
        """Perform backpropagation for parameter gradients.

        Args:
            loss (object): A loss object initiated using any class in the
                "loss" module.

            outputs (list): A list of outputs.

        Returns:
            gradients (dict): the computed gradients of all the layer
                parameters in the graph. See the "__initilize_gradients()"
                function for its structure.

        """

        dv_hs = self.__initialize_memories()
        gradients = self.__initilize_gradients()

        # TODO: Put you code below

        # for i in range(len(outputs) - 1, 0, -1):
        #     delta = loss.backward(outputs[i - 1][-1], gts[i - 1])
        #     dv_x, dv_V, dv_vb = fc.backward(outputs[i - 1][1], delta)
        #     dv_h = dv_x + dv_hs['RNNCell_0']
        #     dv_x, dv_hs['RNNCell_0'], dv_W, dv_U, dv_b = rnn.backward(outputs[i - 1][0], outputs[i - 1][1], outputs[i - 2][1], dv_h)
        #     gradients['FullyConnected_0']['dv_W'] += dv_V
        #     gradients['FullyConnected_0']['dv_b'] += dv_vb
        #     gradients['RNNCell_0']['dv_U'] += dv_U
        #     gradients['RNNCell_0']['dv_W'] += dv_W
        #     gradients['RNNCell_0']['dv_b'] += dv_b

        for t in range(len(outputs) - 1, 0, -1):
            i = len(outputs[t-1]) - 1
            dv_x = loss.backward(outputs[t-1][i], gts[t-1])

            for layer in reversed(self.graph):
                if self.__is_temporal(layer):
                    dv_h = dv_x + dv_hs[layer.name]
                    dv_x, dv_hs[layer.name], dv_W, dv_U, dv_b = layer.backward(outputs[t-1][i - 1], outputs[t-1][i], outputs[t - 2][i], dv_h)
                    gradients[layer.name]["dv_W"] += dv_W
                    gradients[layer.name]["dv_U"] += dv_U
                    gradients[layer.name]["dv_b"] += dv_b
                    i -= 1
                else:
                    dv_x, dv_W, dv_b = layer.backward(np.array(outputs[t-1][i - 1]), dv_x)
                    gradients[layer.name]['dv_W'] += dv_W
                    gradients[layer.name]['dv_b'] += dv_b
                    i -= 1
        return gradients



    def feedforward(self, xs, hprevs):
        """Feed forwad the input sequence through the network.

        This function returns all the outputs (and inputs) of the network
        during feedforward. The data structure of the return value is given
        below.
            outputs = {
                # when time is -1 we use outputs to records all the previous
                # hidden states before feedforward. If a layer does not have
                # a hidden state then its corresponding value is []
                -1: [
                    [],
                    hidden state of layer 0 at time -1,
                                ...
                ],
                0: [
                    input of layer 0 at time 0,
                    output of layer 0 at time 0,
                    output of layer 1 at time 0,
                                ...
                ],
                1: [
                    input of layer 0 at time 1,
                    output of layer 0 at time 1,
                    output of layer 1 at time 1,
                                ...
                ],

                                ...

                m: [
                    input of layer 0 at time n-1,
                    output of layer 0 at time n-1,
                    output of layer 1 at time n-1,
                                ...
                ]
            }

        Args:
            hprevs (dict): the memory states after feedforward. See the
                    "__initialize_memories()" function for its structure.

        Returns:
            outputs (dict): the outputs of the layers in the graph across
                timesteps. See above for its structure.
            hprevs (dict): the memory states after feedforward. See the
                self.__initialize_memories() function for its structure.

        """
        outputs = {
            -1: [[]] + [
                hprevs[layer.name] if self.__is_temporal(layer) else []
                for layer in self.graph
            ]
        }

        # TODO: Put you code below

        for j in range(len(xs)):
            y = xs[j]
            outputs[j] = []
            outputs[j].append(y)

            for i in range(len(self.graph)):
                layer = self.graph[i]

                # check the type of the layer as the forward functions of
                # different layers have different number of parameters
                if self.__is_temporal(layer):
                    y = layer.forward(y, hprevs[layer.name])
                    hprevs[layer.name] = y
                else:
                    y = layer.forward(y)
                outputs[j].append(y)

        return outputs, hprevs

    def sample(self, seed, length, score_function=None):
        """Generate a sample sequence.

        Each generated data at timestep t will be used as the input at timestep
        t + 1.

        Args:
            seed (np.array): the beginning the sequence.
            length (int): the length of the sequence.
            score_function (func): a function that computes probability scores
                for the network outputs.

        Returns:
            sequence (list): the generated sample sequence.

        """
        hprevs = self.__initialize_memories()
        y = seed
        sequence = []
        for t in range(length):
            for i in range(len(self.graph)):
                layer = self.graph[i]

                # check the type of the layer as the forward functions of
                # different layers have different number of parameters
                if self.__is_temporal(layer):
                    y = layer.forward(y, hprevs[layer.name])
                    hprevs[layer.name] = y
                else:
                    y = layer.forward(y)
            if score_function:
                y = score_function(y)
            pred = np.random.choice(range(len(y)), p=y.ravel())
            y = np.zeros(y.shape)
            y[pred] = 1
            sequence.append(pred)
        return sequence

    def train(
            self, training_data, epochs, batch_size, loss, optimizer,
            shuffle=False, reset=False
    ):
        """Train the network.

        Args:
            training_data (list): a list of tuples ``(x, y)`` representing the
                training inputs and the desired outputs.
            epochs (int): number of epochs to train.
            loss (obj): a loss object instantiated using a class from the
                ``loss`` module. The loss function that will be used for the
                training.
            optimizer (obj): a optimizer object instantiated using a class from
                the ``optimization`` module. The optimization method that will
                be used for the training.
            shuffle (bool): if True, shuffle training_data before training
                each epoch
            reset (bool): if True, reset RNN memories after training each data
                sequence. Otherwise, RNN memories will not be reset until the
                entire epoch has been trained.

        """

        print(
            "Training the network for {} epoch(s). "
            "This may take a while.".format(epochs)
        )
        iter_cnt = 0
        seq_length = len(training_data[0][0])
        data_size = len(training_data[0][0][0])
        smooth_loss = -np.log(1.0 / data_size) * seq_length

        for j in xrange(epochs):
            hprevs = self.__initialize_memories()

            if shuffle:
                # shuffle the data to make sure samples in each batch are not
                # correlated
                random.shuffle(training_data)

            n = len(training_data)
            batches = [
                training_data[k:k + batch_size]
                for k in xrange(0, n, batch_size)
            ]

            # mini-batch gradient decent
            for batch in batches:
                batch_gradients = self.__initilize_gradients()

                for xs, gts in batch:
                    # reset RNN memories
                    if reset:
                        hprevs = self.__initialize_memories()

                    # TODO: Put you code below
                    outputs, hprevs = self.feedforward(xs, hprevs)

                    # keep track of the loss
                    total_loss = 0
                    for t in range(len(gts)):
                        total_loss += loss.forward(outputs[t][-1], gts[t])
                    smooth_loss = smooth_loss * 0.999 + total_loss * 0.001
                    if iter_cnt % 100 == 0:
                        print("Iter {}: loss = {}".format(
                            iter_cnt, smooth_loss
                        ))
                    iter_cnt += 1
                    batch_gradients = self.backpropagate(loss, outputs, gts)
                # update gradients at batch level
                optimizer.optimize(self.graph, batch_gradients)
