#Implement Artificial Neural Network training process in Python by using Forward Propagation, Back Propagation.
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the ANN class
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        for i in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[i - 1], layers[i]))

    def forward_propagation(self, X):
        self.activations = [X]
        self.z_values = []
        for i in range(len(self.layers) - 1):
            z = np.dot(self.activations[i], self.weights[i])
            self.z_values.append(z)
            activation = sigmoid(z)
            self.activations.append(activation)
        return self.activations[-1]

    def backward_propagation(self, X, y, learning_rate):
        output = self.forward_propagation(X)
        error = y - output
        delta = error * sigmoid_derivative(output)

        for i in range(len(self.layers) - 2, -1, -1):
            gradient = np.dot(self.activations[i].T, delta)
            self.weights[i] += learning_rate * gradient

            error = np.dot(delta, self.weights[i].T)
            delta = error * sigmoid_derivative(self.activations[i])

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.backward_propagation(X, y, learning_rate)

        return self.forward_propagation(X)

# Testing the ANN
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define the network architecture
layers = [2, 4, 1]

nn = NeuralNetwork(layers)

# Train the network
output = nn.train(X, y, epochs=10000, learning_rate=0.1)

# Print the output after training
print("Output after training:")
print(output)
