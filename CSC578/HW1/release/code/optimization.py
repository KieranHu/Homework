"""All the optimization methods go here.

"""

from __future__ import division, print_function, absolute_import
import random
import numpy as np


class SGD(object):
    """Mini-batch stochastic gradient descent.

    Attributes:
        learning_rate(float): the learning rate to use.
        batch_size(int): the number of samples in a mini-batch.

    """

    def __init__(self, learning_rate, batch_size, decay_rate=1):
        self.learning_rate = float(learning_rate)
        self.batch_size = batch_size
        self.decay_rate = decay_rate

    def __has_parameters(self, layer):
        return hasattr(layer, "W")

    def compute_gradient(self, x, y, graph, loss):
        """ Compute the gradients of network parameters (weights and biases)
        using backpropagation.

        Args:
            x(np.array): the input to the network.
            y(np.array): the ground truth of the input.
            graph(obj): the network structure.
            loss(obj): the loss function for the network.

        Returns:
            dv_Ws(list): a list of gradients of the weights.
            dv_bs(list): a list of gradients of the biases.

        """

        # TODO: Backpropagation code

        # feed forward, compute the result.
        result_for_each_layer = []
        for layer in graph.layers:
            result_for_each_layer.append(x)
            x = layer.forward(x)
        loss_rate = loss.forward(x, y)
        # print('loss_rate', loss_rate)
        # prop back the result from previous result.'

        loss_back = loss.backward(x, y)
        dv_Ws = []  # for each layers
        dv_bs = []  # for each layers
        i = 0
        for layer in graph.layers[::-1]:
            back = layer.backward(result_for_each_layer[::-1][i], loss_back)
            if self.__has_parameters(layer):
                dv_x, dv_W, dv_b = back
                dv_Ws.append(dv_W)
                dv_bs.append(dv_b)
                loss_back = dv_x
            else:
                loss_back = back
            i += 1

        return dv_Ws[::-1], dv_bs[::-1], loss_rate

    def optimize(self, graph, loss, training_data):
        """ Perform SGD on the network defined by 'graph' using
        'training_data'.

        Args:
            graph(obj): a 'Graph' object that defines the structure of a
                neural network.
            loss(obj): the loss function for the network.
            training_data(list): a list of tuples ``(x, y)`` representing the
                training inputs and the desired outputs.

        """

        # Network parameters
        # Ws = [layer.W for layer in graph if self.__has_parameters(layer)]
        # bs = [layer.b for layer in graph if self.__has_parameters(layer)]

        # Shuffle the data to make sure samples in each batch are not
        # correlated
        random.shuffle(training_data)
        n = len(training_data)

        batches = [
            training_data[k:k + self.batch_size]
            for k in xrange(0, n, self.batch_size)
            ]

        # TODO: SGD code
        lmda = 1e-4
        total_loss = []
        batch_loss_rate = []
        for i in batches:
            batch_dW = []
            batch_db = []
            batch_loss = []

            for j in i:
                dv_Ws, dv_bs, loss_rate = self.compute_gradient(np.array(j[0]), np.array(j[1]), graph, loss)
                batch_dW.append(dv_Ws)
                batch_db.append(dv_bs)
                batch_loss.append(loss_rate)
                total_loss.append(loss_rate)

            batch_loss_rate.append(sum(batch_loss)/self.batch_size)

            # print('batch_loss_rate', batch_loss_rate)
            # for lyr in range(len(Ws)):
            #     Ws[lyr] -= self.learning_rate * np.array(sum([k[lyr] for k in batch_dW])) / self.batch_size
            #     bs[lyr] -= self.learning_rate * np.array(sum([k[lyr] for k in batch_db])) / self.batch_size

            i = 0
            for layer in graph:
                if self.__has_parameters(layer):
                    layer.W -= self.learning_rate * (np.array(sum([k[i] for k in batch_dW])) / self.batch_size + lmda * layer.W)
                    layer.b -= self.learning_rate * np.array(sum([k[i] for k in batch_db])) / self.batch_size
                    i += 1
            self.learning_rate *= self.decay_rate
        l2_loss = sum(total_loss) / n
        print('Average l2 Loss:', l2_loss)
        return batch_loss_rate
