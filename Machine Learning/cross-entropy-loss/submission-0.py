import numpy as np

class Solution:
    def binary_cross_entropy(self, y_true: list, y_pred: list) -> float:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Numerical stability: prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary Cross-Entropy formula
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        return np.round(loss, 4)

    def categorical_cross_entropy(self, y_true: list, y_pred: list) -> float:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Numerical stability: prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Categorical Cross-Entropy formula
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        return np.round(loss, 4)