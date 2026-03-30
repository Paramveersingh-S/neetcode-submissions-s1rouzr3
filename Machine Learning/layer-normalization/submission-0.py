import numpy as np

class Solution:
    def forward(self, x: list[float], gamma: list[float], beta: list[float]) -> list[float]:
        x_arr = np.array(x)
        gamma_arr = np.array(gamma)
        beta_arr = np.array(beta)
        epsilon = 1e-5
        
        # 1 & 2. Calculate Mean and Variance across the features
        mu = np.mean(x_arr)
        var = np.var(x_arr)
        
        # 3. Standardize (Center and scale spread)
        x_norm = (x_arr - mu) / np.sqrt(var + epsilon)
        
        # 4. Scale and Shift using learnable parameters
        output = gamma_arr * x_norm + beta_arr
        
        # Return as a standard Python list, rounded to 5 decimal places
        return np.round(output, 5).tolist()