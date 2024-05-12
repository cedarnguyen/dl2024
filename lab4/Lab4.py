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

    def get_weights(self):
        weights = []
        for neuron in self.neurons:
            weights.append(neuron.weights)
        return weights

class Link():
    id = 0
    def __init__(self, fromNeuron, toNeuron, weight=0):
        self.id = f"{self.__class__.__name__}({Link.id})"
        Link.id += 1
        self.fromNeuron = fromNeuron
        self.toNeuron = toNeuron
        self.weight = weight

class LayerLink():
    id = 0
    def __init__(self, fromLayer, toLayer, weights=[]):
        self.id = f"{self.__class__.__name__}({LayerLink.id})"
        LayerLink.id += 1
        self.fromLayer = fromLayer
        self.toLayer = toLayer
        self.links = []
        for i in range(len(fromLayer.neurons)):
            for j in range(len(toLayer.neurons)):
                link = Link(fromLayer.neurons[i], toLayer.neurons[j])
                self.links.append(link)


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

epochs = 10000
learning_rate = 0.1
for epoch in range(epochs):
    for i in range(len(xor_inputs)):
        inputs = xor_inputs[i]
        expected_output = xor_outputs[i]

        hidden_outputs = hidden_layer.forward(inputs)
        output = output_layer.forward(hidden_outputs)

        error = expected_output - output[0]
        output_delta = error * (output[0] * (1 - output[0]))
        hidden_errors = [output_delta * output_layer.neurons[0].weights[j] for j in range(len(hidden_outputs))]
        hidden_deltas = [hidden_errors[j] * (hidden_outputs[j] * (1 - hidden_outputs[j])) for j in range(len(hidden_outputs))]

        for j in range(len(hidden_layer.neurons)):
            hidden_layer.neurons[j].weights = [hidden_layer.neurons[j].weights[k] + (learning_rate * hidden_deltas[j] * inputs[k]) for k in range(len(inputs))]
            hidden_layer.neurons[j].bias += learning_rate * hidden_deltas[j]

        output_layer.neurons[0].weights = [output_layer.neurons[0].weights[j] + (learning_rate * output_delta * hidden_outputs[j]) for j in range(len(hidden_outputs))]
        output_layer.neurons[0].bias += learning_rate * output_delta

for i in range(len(xor_inputs)):

    
    inputs = xor_inputs[i]
    hidden_outputs = hidden_layer.forward(inputs)
    output = output_layer.forward(hidden_outputs)
    print(f"Input: {inputs}, Output: {output[0]}")
