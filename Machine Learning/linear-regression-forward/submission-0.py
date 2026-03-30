import numpy as np

class Solution:
    def get_model_prediction(self, X: list[list[float]], weights: list[float]) -> list[float]:
        # Convert lists to NumPy arrays for vectorized matrix math
        X_arr = np.array(X)
        W_arr = np.array(weights)
        
        # Compute the dot product: X * W
        predictions = np.dot(X_arr, W_arr)
        
        # Return as a standard Python list, rounded to 5 decimal places
        return np.round(predictions, 5).tolist()

    def get_error(self, model_prediction: list[float], ground_truth: list[float]) -> float:
        preds = np.array(model_prediction)
        truth = np.array(ground_truth)
        
        # Compute Mean Squared Error
        mse = np.mean((preds - truth) ** 2)
        
        return round(float(mse), 5)