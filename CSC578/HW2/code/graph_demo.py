"""This demo shows some usages of the Graph class. You do not have to waste
your time trying to figure out how the Graph class is implemented.

"""

from graph import Graph

graph_config = [
    ("Conv2D", {"filter_size": (64, 1, 3, 3), "strides": 1, "padding": 1}),
    ("ReLU", {}),
    ("Conv2D", {"filter_size": (32, 64, 3, 3), "strides": 1, "padding": 1}),
    ("ReLU", {}),
    ("FullyConnected", {"shape": (10, 25088)})
]

graph = Graph(graph_config)

# Look at the graph structure
print(graph)
print("\n-----\n")
"""
Conv2D_0((64, 1, 3, 3), (1, 1), (1, 1))
ReLU_0
Conv2D_1((32, 64, 3, 3), (1, 1), (1, 1))
ReLU_1
FullyConnected_0(10, 25088)
"""

# Access by name
print(graph["Conv2D_1"])
print("\n-----\n")
"""
Conv2D_1((32, 64, 3, 3), (1, 1), (1, 1))
"""

# Access by index
print(graph[3])
print("\n-----\n")
"""
ReLU_1
"""

# Iterate through for loop
for layer in graph:
    print(layer.name)
print("\n-----\n")
"""
Conv2D_0
ReLU_0
Conv2D_1
ReLU_1
FullyConnected_0
"""

# get the size of the graph
print(len(graph))
print("\n-----\n")
"""
5
"""
