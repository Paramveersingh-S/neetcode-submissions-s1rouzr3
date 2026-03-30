import numpy as np

class Solution:
    def forward(self, x: list[float], w: list[float], b: float, activation: str) -> float:
        # Convert lists to NumPy arrays for vectorized math
        x_arr = np.array(x)
        w_arr = np.array(w)
        
        # Step 1 & 2: Calculate the dot product and add the bias
        z = np.dot(w_arr, x_arr) + b
        
        # Step 3: Apply the chosen activation function
        if activation == "sigmoid":
            output = 1 / (1 + np.exp(-z))
        elif activation == "relu":
            output = np.maximum(0, z)
            
        # Return the final output rounded to 5 decimal places
        return float(np.round(output, 5))