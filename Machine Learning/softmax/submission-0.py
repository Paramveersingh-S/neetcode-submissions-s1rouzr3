import numpy as np

class Solution:
    def softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z))
        return np.round(exp_z / np.sum(exp_z), 4)