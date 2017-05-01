"""All the loss functions go here.

"""

from __future__ import division, print_function, absolute_import

import numpy as np


class LogSoftmax(object):
    """The log softmax (sparse) loss 'L = -log(softmax(y_gt))'.

    """
    def __init__(self, name="LogSoftmax"):
        self.name = name

    def forward(self, y, gt):
        """Compute the log softmax loss.

        Args:
            y (np.array): the output from previous layer. It is a column
                vector.
            gt (int): the ground truth. Note it is an integer indicate the
                label of an input data.

        Return:
            The log softmax loss.

        """

        # TODO: Put you code below
        score = np.log(np.exp(y) / np.sum(np.exp(y)))
        temp = np.zeros([len(y), 1])
        temp[gt - 1, :] = 1
        return -np.sum(temp * score)

    def backward(self, y, gt):
        """Compute the derivative of the log softmax loss.

        Args:
            y (np.array): the output from previous layer. It is a column
                vector.
            gt (int): the ground truth. Note it is an integer indicate the
                label of an input data.

        Returns:
            The derivative of the loss with respect to the y.

        """

        # TODO: Put you code below
        temp = np.zeros([len(y), 1])
        temp[gt - 1, :] = 1
        p = np.exp(y) / np.sum(np.exp(y))
        return p - temp
