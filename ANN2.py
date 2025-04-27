# Generate ANDNOT function using McCulloch-Pitts neural net by a python  program. 

import numpy as np

def mcculloch_pitts_andnot(a, b):
    # Define weights and threshold
    # A has weight 1, NOT B has weight -1
    weights = np.array([1, -1])  
    threshold = 1 
    
    # Compute weighted sum
    weighted_sum = a * weights[0] + b * weights[1]
    
    # Apply activation function (Step function)
    return 1 if weighted_sum >= threshold else 0

#main code
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
print("A B | A ANDNOT B")
print("---------------")
for a, b in inputs:
    output = mcculloch_pitts_andnot(a, b)
    print(f"{a} {b} | {output}")
