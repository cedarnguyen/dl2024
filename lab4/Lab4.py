import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Neuron():
    id = 0
    def __init__(self, activation=sigmoid):
        self.id = f"{self.__class__.__name__}({Neuron.id})"
        Neuron.id += 1
        self.activation = activation
        self.output = 0
        self.weights = [] 
        self.bias = 1      

    def linearsum(self, weights, inputs):
        return sum([weights[i] * inputs[i] for i in range(len(weights))])

    def activate(self, z):
        return self.activation(z)

    def forward(self, inputs):
        z = self.linearsum(self.weights, inputs) + self.bias
        self.output = self.activate(z)
        return self.output

class Layer():
    id = 0
    def __init__(self, nodes, activation=sigmoid):
        self.id = f"{self.__class__.__name__}({Layer.id})"
        Layer.id += 1
        self.neurons = [Neuron(activation) for _ in range(nodes)]

    def connect(self, prev_layer):
        for neuron in self.neurons:
            neuron.weights = [random.uniform(-1, 1) for _ in range(len(prev_layer.neurons))]

    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(inputs))
        return outputs

xor_inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
xor_outputs = [0, 1, 1, 0]

input_layer = Layer(2)
hidden_layer = Layer(2)
output_layer = Layer(1)

input_layer.connect(input_layer)  
hidden_layer.connect(input_layer)
output_layer.connect(hidden_layer)

for i in range(len(xor_inputs)):
    inputs = xor_inputs[i]
    hidden_outputs = hidden_layer.forward(inputs)
    output = output_layer.forward(hidden_outputs)
    print(f"Input: {inputs}, Output: {output[0]}")
    