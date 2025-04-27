#Write a Python Program using Perceptron Neural Network to recognise even and odd numbers. Given numbers are in ASCII form 0 to 9

import numpy as np

def prediction(x):
    return 1 if x >= 0 else 0

# Input from user
num = int(input("ENTER A DIGIT FROM (0-9):"))

# Correct training data
# Inputs: 6-bit binary representation
# Labels: 1 = odd, 0 = even
training_data = [
    {'input': [0,0,0,0,0,0], 'label': 0},  # 0 even
    {'input': [0,0,0,0,0,1], 'label': 1},  # 1 odd
    {'input': [0,0,0,0,1,0], 'label': 0},  # 2 even
    {'input': [0,0,0,0,1,1], 'label': 1},  # 3 odd
    {'input': [0,0,0,1,0,0], 'label': 0},  # 4 even
    {'input': [0,0,0,1,0,1], 'label': 1},  # 5 odd
    {'input': [0,0,0,1,1,0], 'label': 0},  # 6 even
    {'input': [0,0,0,1,1,1], 'label': 1},  # 7 odd
    {'input': [0,0,1,0,0,0], 'label': 0},  # 8 even
    {'input': [0,0,1,0,0,1], 'label': 1}   # 9 odd
]

# Initialize weights
weights = np.zeros(6)
learning_rate = 0.1

# Training
for epoch in range(100):
    for data in training_data:
        inputs = np.array(data['input'])
        label = data['label']
        output = prediction(np.dot(inputs, weights))
        error = label - output
        weights += learning_rate * error * inputs

# User input conversion
user_input = [int(bit) for bit in f'{num:06b}']
print("Binary input:", user_input)

out = prediction(np.dot(user_input, weights))

if out == 0:
    print('EVEN number')
else:
    print('ODD number')
