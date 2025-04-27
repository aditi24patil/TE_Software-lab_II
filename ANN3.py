#Write a Python Program using Perceptron Neural Network to recognise even and odd numbers. Given numbers are in ASCII form 0 to 9

import numpy as np

def ascii_to_binary_array(ch):
    return np.array([int(bit) for bit in format(ord(ch), '08b')])

training_data = []
labels = []

for i in range(10):
    ascii_array = ascii_to_binary_array(str(i))
    training_data.append(ascii_array)
    if i % 2 == 0:
        labels.append(0)  # even
    else:
        labels.append(1)  # odd

training_data = np.array(training_data)
labels = np.array(labels)

weights = np.zeros(8)
bias = 0
learning_rate = 0.1

def step_function(x):
    return 1 if x >= 0 else 0

for epoch in range(100):
    total_error = 0
    for inputs, label in zip(training_data, labels):
        linear_output = np.dot(inputs, weights) + bias
        prediction = step_function(linear_output)
        error = label - prediction
        weights += learning_rate * error * inputs
        bias += learning_rate * error
        total_error += abs(error)
    if total_error == 0:
        break

print("\nTraining complete!")

while True:
    ch = input("\nEnter a single digit (0-9) or 'q' to quit: ")
    if ch == 'q':
        break
    if ch not in "0123456789":
        print("Invalid input! Enter a single digit.")
        continue

    input_array = ascii_to_binary_array(ch)
    output = step_function(np.dot(input_array, weights) + bias)

    if output == 0:
        print(f"{ch} is Even!")
    else:
        print(f"{ch} is Odd!")
******************************************************************************************************************************************************************************
OR other option
