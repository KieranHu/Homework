import cPickle
import gzip
import numpy as np


def load_tinyshakespeare(
        input_file="../data/tinyshakespeare/input.txt",
        sequence_size=25
):
    """

    Args:
        input_file (str): a plain text file containing a subset of works of
            Shakespeare.

    Returns:
        training_data (list): loaded Shakespeare's works.
        vocab (list): a list of vocabulary used in Shakespeare's works.
    """
    data = open(input_file, 'r').read()
    vocab = sorted(list(set(data)))

    data_size, vocab_size = len(data), len(vocab)
    vocab2index = {v: i for i, v in enumerate(vocab)}

    training_data = []
    for i in range(0, len(data), sequence_size)[:-1]:
        xs = []
        ys = []
        if i + sequence_size >= data_size:
            break
        for j in range(sequence_size):
            x_char = data[i + j]
            x = np.zeros((vocab_size, 1))
            x[vocab2index[x_char]] = 1
            xs.append(x)

            y_char = data[i + j + 1]
            y = vocab2index[y_char]
            ys.append(y)
        training_data.append((xs, ys))

    return training_data, vocab


def load_raw_mnist():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.

    """

    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)


def load_mnist(vectorize_image=True, vectorize_label=True):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code.

    """

    tr_d, va_d, te_d = load_raw_mnist()
    training_inputs = [
        np.reshape(x, (784, 1)) if vectorize_image else x
        for x in tr_d[0]
    ]
    training_results = [
        vectorized_result(y) if vectorize_image else y for y in tr_d[1]
    ]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [
        np.reshape(x, (784, 1)) if vectorize_image else x
        for x in va_d[0]
    ]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [
        np.reshape(x, (784, 1)) if vectorize_image else x
        for x in te_d[0]
    ]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network.

    """

    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
