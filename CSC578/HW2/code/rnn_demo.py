"""This is a simple example for training a recurrent neural network. This
example demonstrates some usage of the modules. You may use this code to test
your implementation.

Goal: this demo tries to learn the writing style from the training data (i.e.
Shakespeare's works).

Loss: if you code is well-implemented, you will see a loss of about 55 after
training for 6000 iterations.

Sample: the generated sample script may still not readable due to the
simplicity of the RNN structure. But your RNN should at least be able to learn
somewhat meaningful patterns. Here is an example of generated script:

-------
he sorcol to thichted!

ALANE:
Youl'd ned y us, COdre, mose
Hot se I krarse--
To fyouees uns, balon the thendd ele, ins to thiy Cfracing of Arer, ul tI
statof, ur her, poe.

ELAMANLA:
Coe mas of han f
-------

"""

import data
import numpy as np
from graph import Graph
from initializer import Normal, Zeros
from loss import LogSoftmax
from network import RNN
from optimization import Adagrad

# Load the tinyshakespear dataset
training_data, vocab = data.load_tinyshakespeare()
vocab_size = len(vocab)

# The network definition of a neural network
graph_config = [
    (
        "RNNCell",
        {
            "shape": (100, vocab_size),
            "weights_init": Normal(0, 0.1),
            "bias_init": Zeros()
        }
    ),
    (
        "FullyConnected",
        {
            "shape": (vocab_size, 100),
            "weights_init": Normal(0, 0.1),
            "bias_init": Zeros()
        }
    )
]

graph = Graph(graph_config)
loss = LogSoftmax()
optimizer = Adagrad(0.1)  # learning rate 0.1
rnn = RNN(graph)

rnn.train(training_data, 1, 1, loss, optimizer)

# Generate sample scripts
seed = np.zeros((vocab_size, 1))
seed[np.random.randint(vocab_size)] = 1
sample = rnn.sample(seed, 200, lambda x: np.exp(x) / np.sum(np.exp(x)))
sample_script = "".join([vocab[s] for s in sample])
print("----------\n{}\n----------".format(sample_script))
