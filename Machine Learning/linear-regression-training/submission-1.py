import numpy as np

class Solution:
    def train_model(self, X: list[list[float]], Y: list[float], num_iterations: int, initial_weights: list[float]) -> list[float]:
        X_arr = np.array(X)
        Y_arr = np.array(Y)
        weights = np.array(initial_weights, dtype=float)
        
        # Using a fallback just in case the platform also forgot to inject self.learning_rate
        learning_rate = getattr(self, 'learning_rate', 0.01)
        N = len(Y_arr)
        
        for _ in range(num_iterations):
            # 1. Forward Pass
            predictions = np.dot(X_arr, weights)
            
            # 2. Compute Gradients manually (bypassing the missing get_derivative)
            error = predictions - Y_arr
            gradients = (2 / N) * np.dot(X_arr.T, error)
            
            # 3. Update Weights
            weights -= learning_rate * gradients
            
        # The prompt asks for a NumPy array, but the example shows a standard list. 
        # .tolist() ensures it formats exactly like the expected output.
        return np.round(weights, 5).tolist()