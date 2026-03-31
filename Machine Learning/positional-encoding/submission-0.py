import numpy as np

class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> np.ndarray:
        # 1. Initialize an empty matrix of shape (seq_len, d_model)
        pe = np.zeros((seq_len, d_model))
        
        # 2. Create a column vector of positions: [[0], [1], [2], ..., [seq_len-1]]
        position = np.arange(seq_len)[:, np.newaxis]
        
        # 3. Calculate the division term for the frequencies
        # We only need to compute this for the even indices (0, 2, 4, ...)
        even_indices = np.arange(0, d_model, 2)
        div_term = 1 / np.power(10000, even_indices / d_model)
        
        # 4. Apply Sine to all even columns (0, 2, 4, ...)
        pe[:, 0::2] = np.sin(position * div_term)
        
        # 5. Apply Cosine to all odd columns (1, 3, 5, ...)
        pe[:, 1::2] = np.cos(position * div_term)
        
        # 6. Round to 5 decimal places as requested
        return np.round(pe, 5)