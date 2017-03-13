"""This is a simple example for training a neural network. This example
demonstrates some usage of the modules. You may use this code to test your
implementation.

"""

import mnist_loader
import numpy as np
import matplotlib.pyplot as plt
from graph import Graph
from loss import Euclidean
from network import Network
from optimization import SGD

# Load the MNIST dataset
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# The network definition of a neural network
graph_config = [
    ("FullyConnected", {"shape": (30, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 30)}),
    ("Sigmoid", {})
]

graph = Graph(graph_config)
loss = Euclidean()
optimizer = SGD(learning_rate=1, batch_size=10)  # learning rate 3.0, batch size 10
network = Network(graph)

# Train the network for 1 epoch
batch_loss_rate = network.train(training_data, 5, loss, optimizer, test_data)

# Test a handwritten digit image
x, y = test_data[1]
y_pred = np.argmax(network.inference(x))
print("The image is {}. Your prediction is {}.".format(y, y_pred))

# without decay
count = range(len(batch_loss_rate))
line = plt.plot(count, batch_loss_rate)
plt.xlabel('count')
plt.ylabel('loss')
# plt.text(60, 2, r'leanring rate = 1, zero initial')
plt.setp(line, color='y', linewidth=0.5)
plt.show()
