"""All the initializers go here.

"""

from __future__ import division, print_function, absolute_import
import numpy as np


class Normal(object):
    """Initialization with random values from a normal distribution.

    Args:
        mean (float): the mean of the normal distribution.
        stddev (float): the standard deviation of the normal distribution.

    Attributes:
        mean (float): the mean of the normal distribution.
        stddev (float): the standard deviation of the normal distribution.

    """

    def __init__(self, mean=0.0, stddev=0.1):
        self.mean = mean
        self.stddev = stddev

    def initialize(self, shape):
        return self.stddev ** 2 * np.random.randn(*shape) + self.mean


class Zeros(object):
    """Initialization with all elements set to zero.

    """

    def initialize(self, shape):
        return np.zeros(shape)
