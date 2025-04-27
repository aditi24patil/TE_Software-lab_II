#With a suitable example demonstrate the perceptron learning law with its decision regions using python. Give the output in graphical form
import numpy as np
import matplotlib.pyplot as plt

# Generate a linearly separable dataset
np.random.seed(1)
X = np.array([[2, 3], [1, 1], [4, 2], [2, 2], [3, 1], [5, 3]])
y = np.array([1, -1, 1, -1, -1, 1])  # Class labels

# Step 2: Initialize weights and bias
w = np.zeros(2)
b = 0
learning_rate = 0.1

# Perceptron Learning Algorithm
def perceptron_train(X, y, w, b, lr, epochs=10):
    for epoch in range(epochs):
        for i in range(len(X)):
            activation = np.dot(X[i], w) + b
            predicted = 1 if activation >= 0 else -1
            
            # Weight update rule if misclassified
            if predicted != y[i]:
                w += lr * y[i] * X[i]
                b += lr * y[i]
    
    return w, b

w, b = perceptron_train(X, y, w, b, learning_rate)

# Visualizing the decision boundary
def plot_decision_boundary(X, y, w, b):
    plt.figure(figsize=(6, 5))
    
    # Plot data points
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(X[i][0], X[i][1], marker="o", color="blue", label="Class 1" if i == 0 else "")
        else:
            plt.scatter(X[i][0], X[i][1], marker="s", color="red", label="Class -1" if i == 1 else "")

    # Decision boundary: w1*x + w2*y + b = 0 => y = -(w1/w2) * x - b/w2
    x_vals = np.linspace(0, 6, 100)
    y_vals = -(w[0] / w[1]) * x_vals - b / w[1]
    plt.plot(x_vals, y_vals, 'g--', label="Decision Boundary")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Perceptron Decision Boundary")
    plt.grid(True)
    plt.show()

plot_decision_boundary(X, y, w, b)
