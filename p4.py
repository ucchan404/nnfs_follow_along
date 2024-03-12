import numpy as np

inputs = [[1, 2, 3, 2.5], #a batch of inputs, ie inputs from several neurons in the same layer
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

#a second layer of neurons with their own weights and biases
weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.1291, -0.33],
           [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_output = np.dot(inputs, np.array(weights).T) + biases

layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2 # second layer uses the outputs from the first layer as input

# print(layer2_output)


#convert concept of layers into objects

np.random.seed(0) # used to get the same numbers as in video

X = [[1, 2, 3, 2.5], #name for input/feature set/training data set is denoted by X as a standard
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #multiplying by 0.10 to get the weights to be in the range of (-1,1).
                                                                  # We do inputs then neurons to avoid having to transpose all the time
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5,2) #input onlayer 2 must be the same size as output from layer 1

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
