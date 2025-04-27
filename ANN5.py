#Write a python Program for Bidirectional Associative Memory with two pairs of vectors

import numpy as np

def signum(vector):
    return np.where(vector >= 0, 1, -1)

def train_bam(X, Y):
    W = np.zeros((X.shape[1], Y.shape[1]))
    for i in range(len(X)):
        W += np.outer(X[i], Y[i])
    return W

def recall_bam(W, input_pattern, direction="forward"):
    if direction == "forward":
        return signum(np.dot(W.T, input_pattern))
    elif direction == "backward":
        return signum(np.dot(W, input_pattern))

# Define input-output training pairs (bipolar vectors)
X = np.array([[1, -1, 1], [-1, 1, -1]])
Y = np.array([[1, 1, -1], [-1, -1, 1]])

# Train BAM
W = train_bam(X, Y)

# Test recall in both directions
test_X = X[0]
test_Y = recall_bam(W, test_X, "forward")
recalled_X = recall_bam(W, test_Y, "backward")

print("Original X:", test_X)
print("Recalled Y:", test_Y)
print("Recalled X:", recalled_X)
