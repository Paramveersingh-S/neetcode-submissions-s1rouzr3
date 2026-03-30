import numpy as np

class Solution:
    def backward(self, x: list[float], w: list[float], b: float, y_true: float) -> tuple[list[float], float]:
        x_arr = np.array(x)
        w_arr = np.array(w)
        
        # 1. Forward Pass
        z = np.dot(w_arr, x_arr) + b
        y_pred = 1 / (1 + np.exp(-z))
        
        # 2. Backward Pass (Chain Rule)
        # Derivative of Loss w.r.t prediction
        dL_dy = y_pred - y_true
        
        # Derivative of Sigmoid w.r.t raw output z
        dy_dz = y_pred * (1 - y_pred)
        
        # Local error at the neuron (delta)
        dL_dz = dL_dy * dy_dz
        
        # Gradients for weights and bias
        dL_dw = dL_dz * x_arr
        dL_db = dL_dz
        
        # Return tuple with 5 decimal place precision
        return (np.round(dL_dw, 5).tolist(), round(float(dL_db), 5))