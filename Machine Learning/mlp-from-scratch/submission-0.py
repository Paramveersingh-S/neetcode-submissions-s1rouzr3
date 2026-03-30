import numpy as np

class Solution:
    def forward(self, x: list[float], weights: list[list[list[float]]], biases: list[list[float]]) -> np.ndarray:
        # Initialize the activation as the input vector
        a = np.array(x)
        
        num_layers = len(weights)
        
        for i in range(num_layers):
            W = np.array(weights[i])
            b = np.array(biases[i])
            
            # Linear Transformation: z = a * W + b
            z = np.dot(a, W) + b
            
            # Apply ReLU activation for all hidden layers
            if i < num_layers - 1:
                a = np.maximum(0, z)
            else:
                a = z  # No activation on the output layer
                
        # Return the final output rounded to 5 decimal places
        return np.round(a, 5)