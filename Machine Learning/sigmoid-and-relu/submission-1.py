import numpy as np

class Solution:
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return np.round(1 / (1 + np.exp(-z)), 5)
        
    def relu(self, z: np.ndarray) -> np.ndarray:
        return np.round(np.maximum(0, z), 5)