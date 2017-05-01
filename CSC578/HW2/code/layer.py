"""All the layer functions go here.

"""

from __future__ import print_function, absolute_import
import numpy as np


class FullyConnected(object):
    """Fully connected layer 'y = Wx + b'.

    Arguments:
        shape (tuple): the shape of the fully connected layer. shape[0] is the
            output size and shape[1] is the input size.
        weights_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        bias_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        name (str): the name of the layer.

    Attributes:
        W (np.array): the weights of the fully connected layer.
        b (np.array): the biases of the fully connected layer.
        shape (tuple): the shape of the fully connected layer. shape[0] is the
            output size and shape[1] is the input size.
        name (str): the name of the layer.

    """

    def __init__(
        self, shape, weights_init=None, bias_init=None, name="FullyConnected"
    ):
        self.W = weights_init.initialize(shape) \
            if weights_init else np.random.randn(*shape)
        self.b = bias_init.initialize((shape[0], 1)) \
            if bias_init else np.random.randn(shape[0], 1)
        self.shape = shape
        self.name = name

    def __repr__(self):
        return "{}({}, {})".format(self.name, self.shape[0], self.shape[1])

    def forward(self, x):
        """Compute the layer output.

        Args:
            x (np.array): the input of the layer.

        Returns:
            The output of the layer.

        """

        Y = np.dot(self.W, x) + self.b
        return Y

    def backward(self, x, dv_y):
        """Compute the gradients of weights and biases and the gradient with
        respect to the input.

        Args:
            x (np.array): the input of the layer.
            dv_y (np.array): The derivative of the loss with respect to the
                output.

        Returns:
            dv_x (np.array): The derivative of the loss with respect to the
                input.
            dv_W (np.array): The derivative of the loss with respect to the
                weights.
            dv_b (np.array): The derivative of the loss with respect to the
                biases.

        """

        dv_x = np.dot(self.W.T, dv_y)
        dv_W = np.dot(dv_y, x.T)
        dv_b = dv_y

        return dv_x, dv_W, dv_b


class Conv2D(object):
    """2D convolutional layer.

    Arguments:
        filter_size (tuple): the shape of the filter. It is a tuple = (
            out_channels, in_channels, filter_height, filter_width).
        strides (int or tuple): the strides of the convolution operation.
            padding (int or tuple): number of zero paddings.
        weights_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        bias_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        name (str): the name of the layer.

    Attributes:
        W (np.array): the weights of the layer. A 4D array of shape (
            out_channels, in_channels, filter_height, filter_width).
        b (np.array): the biases of the layer. A 1D array of shape (
            in_channels).
        filter_size (tuple): the shape of the filter. It is a tuple = (
            out_channels, in_channels, filter_height, filter_width).
        strides (tuple): the strides of the convolution operation. A tuple = (
            height_stride, width_stride).
        padding (tuple): the number of zero paddings along the height and
            width. A tuple = (height_padding, width_padding).
        name (str): the name of the layer.

    """

    def __init__(
            self, filter_size, strides, padding,
            weights_init=None, bias_init=None, name="Conv2D"
    ):
        self.W = weights_init.initialize(filter_size) \
            if weights_init else np.random.randn(*filter_size)
        self.b = bias_init.initialize((filter_size[0], 1)) \
            if bias_init else np.random.randn(filter_size[0], 1)
        self.filter_size = filter_size
        self.strides = (strides, strides) if type(strides) == int else strides
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
            self.name, self.filter_size, self.strides, self.padding
        )

    def forward(self, x):
        """Compute the layer output.

        Args:
            x (np.array): the input of the layer. A 3D array of shape (
                in_channels, in_heights, in_weights).

        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).

        """
        p, s = self.padding, self.strides
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.W.shape[2] + 2 * p[0]) % s[0] == 0, \
            'Width does not work'
        assert (x.shape[2] - self.W.shape[3] + 2 * p[1]) % s[1] == 0, \
            'Height does not work'

        # TODO: Put you code below
        # output K, H, W
        K = self.W.shape[0]
        H = (x.shape[1] - self.filter_size[2] + 2 * p[0])/s[0] + 1
        W = (x.shape[2] - self.filter_size[3] + 2 * p[1])/s[1] + 1
        y = np.empty([K, H, W])

        for k in range(K):
            for i in range(H):
                for j in range(W):
                    y[k][i][j] += sum(x_padded[t][i * s[0] + m][j * s[1] + n] * self.W[k][t][m][n] \
                        for t in range(self.filter_size[1]) for m in range(self.filter_size[2]) for n in range(self.filter_size[3])) + self.b[k]
        return y

    def backward(self, x, dv_y):
        """Compute the gradients of weights and biases and the gradient with
        respect to the input.

        Args:
            x (np.array): the input of the layer. A 3D array of shape (
                in_channels, in_heights, in_weights).
            dv_y (np.array): The derivative of the loss with respect to the
                output. A 3D array of shape (out_channels, out_heights,
                out_weights).

        Returns:
            dv_x (np.array): The derivative of the loss with respect to the
                input. It has the same shape as x.
            dv_W (np.array): The derivative of the loss with respect to the
                weights. It has the same shape as self.W
            dv_b (np.array): The derivative of the loss with respect to the
                biases. It has the same shape as self.b

        """
        p, s = self.padding, self.strides
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # TODO: Put you code below
        K = self.W.shape[0]
        H = (x.shape[1] - self.filter_size[2] + 2 * p[0])/s[0] + 1
        W = (x.shape[2] - self.filter_size[3] + 2 * p[1])/s[1] + 1
        T = self.filter_size[1]
        M = self.filter_size[2]
        N = self.filter_size[3]

        dv_x = np.zeros(x.shape)
        dv_W = np.zeros(self.W.shape)
        dv_b = np.zeros(self.b.shape)

        for k in range(K):
            for t in range(T):
                for m in range(M):
                    for n in range(N):
                        dv_W[k][t][m][n] += sum(dv_y[k][i][j]*x_padded[t][i*s[0]+m][j*s[1]+n] \
                                                for i in range(H) for j in range(W))
            dv_b[k] = sum(sum(dv_y[k]))

        for k in range(K):
            for t in range(T):
                for u in range(x.shape[1]):
                    for v in range(x.shape[2]):
                        dv_x[t][u][v] += sum(dv_y[k][i][j] * self.W[k][t][m][n] \
                                             for m in range(M) \
                                             for n in range(N) \
                                             for i in range(H) \
                                             for j in range(W) if i*s[0]+m == u + p[0] and j*s[1]+n == v + p[0])

        return dv_x, dv_W, dv_b



class RNNCell(object):
    """A vanilla RNN cell. All the computations are only for one timestep.

    z_t = W * x_t + U * h_{t-1} + b
    h_t = tanh(z_t)

    Args:
        shape (tuple): shape[0], hidden_size; shape[1], input_size
        weights_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        bias_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        name (str): the name of the layer.

    Attributes:
        W (np.array): the weights with respect to input.
        U (np.array): the weights with respect to hidden state.
        b (np.array): the bias parameter.

    """

    def __init__(
        self, shape, weights_init=None, bias_init=None, name="RNNCell"
    ):
        self.W = weights_init.initialize(shape) \
            if weights_init else np.random.randn(*shape)
        self.U = weights_init.initialize((shape[0], shape[0])) \
            if weights_init else np.random.randn(shape[0], shape[0])
        self.b = bias_init.initialize((shape[0], 1)) \
            if bias_init else np.random.randn(shape[0], 1)
        self.shape = shape
        self.name = name

    def __repr__(self):
        return "{}({}, {})".format(self.name, self.shape[0], self.shape[1])

    def forward(self, x, hprev):
        """Run a forward pass for one timestep.

        Args:
            x (np.array): input data. A column vector of size (shape[1], 1).
            hprev (np.array): hidden state t - 1. A column vector of size
                (shape[0], 1).

        Returns:
            Updated hidden state. A column vector of size (shape[0], 1).

        """

        # TODO: Put you code below
        return np.tanh(self.W.dot(x) + self.U.dot(hprev) + self.b)

    def backward(self, x, h, hprev, dv_h):
        """Run a backward pass for one timestep.

        Args:
            x (np.array): input data. A column vector of size (shape[1], 1).
            h (np.array): hidden state at t. A column vector of size
                (shape[0], 1).
            hprev (np.array): hidden state at t - 1. A column vector of size
                (shape[0], 1).
            dv_h (np.array): the derivative of loss with respect to the
                current hidden state. A column vector of size (shape[0], 1).

        Returns:
            dv_x (np.array): the derivative of loss with respect to input. A
                column vector of size (shape[1], 1).
            dv_hprev (np.array): the derivative of loss with respect to
                hidden state at t - 1. A column vector of size (shape[0], 1).
            dv_W (np.array): the derivative of loss with respect to W. A
                matrix of size(shape[0], shape[1]).
            dv_U (np.array): the derivative of loss with respect to U. A
                matrix of size(shape[0], shape[0]).
            dv_b (np.array): the derivative of loss with respect to bias. A
                column vector of size(shape[0], 1).

        """

        # TODO: Put you code below
        dv_b = 1 - pow(h, 2)
        delta = dv_h * dv_b
        dv_x = self.W.T.dot(delta)
        dv_hprev = self.U.T.dot(delta)
        dv_W = delta.dot(x.T)
        dv_U = delta.dot(hprev.T)
        return dv_x, dv_hprev, dv_W, dv_U, dv_b

class ReLU(object):
    """Rectified Linear Unit "y = max(0, x)"

    Args:
        name (str): the name of the layer.

    Attributes:
        name (str): the name of the layer.

    """

    def __init__(self, name="ReLU"):
        self.name = name

    def __repr__(self):
        return "{}".format(self.name)

    def forward(self, x):
        """Compute the layer output.

            Args:
                x (np.array): the input of the layer.

            Returns:
                The output of the layer.

        """

        return np.maximum(0, x)

    def backward(self, x, dv_y):
        """Compute the gradient with respect to the input.

        Args:
            x (np.array): the input of the layer.
            dv_y (np.array): The derivative of the loss with respect to the
                output.

        Returns:
            The derivative of the loss with respect to the input.

        """

        dv_x = np.copy(dv_y)
        dv_x[x <= 0] = 0
        return dv_x
