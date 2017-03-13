from __future__ import division, print_function, absolute_import
import numpy as np


class Network(object):
    """Neural Network.

    Attributes:
        graph(obj): A "Graph" object that describes the layers of the network.

    """

    def __init__(self, graph):
        self.graph = graph

    def inference(self, a):
        """Feedforward an input to the network.

        Args:
            a(np.array): the input data.

        Returns:
            The output of the network.
        """

        for layer in self.graph:
            a = layer.forward(a)
        return a

    def train(self, training_data, epochs, loss, optimizer, test_data=None):
        """Train the neural network.

        Args:
            training_data(list): a list of tuples ``(x, y)`` representing the
                training inputs and the desired outputs.
            epochs(int): number of epochs to train.
            loss(obj): a loss object instantiated using a class from the
                ``loss`` module. The loss function that will be used for the
                training.
            optimizer(obj): a optimizer object instantiated using a class from
                the ``optimization`` module. The optimization method that will
                be used for the training.
            test_data(list, optional): a list of tuples ``(x, y)`` representing
                the inputs and the desired outputs. If provided, then the
                network will be evaluated against the test data after each
                epoch, and partial progress printed out.  This is useful for
                tracking progress, but slows things down substantially.

        """

        print(
            "Training the network for {} epoch(s). "
            "This may take a while.".format(epochs)
        )
        lst =[]
        for j in xrange(epochs):
            batch_loss_rate = optimizer.optimize(self.graph, loss, training_data)

            if test_data:
                print(
                    "Epoch {0}: {1} / {2}".format(
                        j, self.test(test_data), len(test_data)
                    )
                )
            else:
                print("Epoch {0} complete".format(j))

            lst.extend(batch_loss_rate)

        return lst

    def test(self, test_data):
        """Test the network

        Args:
            test_data(list): a list of tuples ``(x, y)`` representing the
            inputs and the desired outputs. If provided, then the network will
            be evaluated against the test data after each epoch, and partial
            progress printed out.  This is useful for tracking progress, but
            slows things down substantially.

        Returns:
            The number of test inputs for which the neural network outputs the
            correct result. Note that the neural network's output is assumed to
            be the index of whichever neuron in the final layer has the highest
            activation.

        """
        test_results = [(np.argmax(self.inference(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
