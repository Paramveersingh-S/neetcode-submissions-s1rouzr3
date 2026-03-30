import torch
import torch.nn.functional as F

class Solution:
    def reshape(self, to_reshape: list[list[float]]) -> list[list[float]]:
        t = torch.tensor(to_reshape)
        # -1 infers the correct number of rows automatically
        reshaped = t.reshape(-1, 2)
        return reshaped.tolist()

    def average(self, to_avg: list[list[float]]) -> list[float]:
        t = torch.tensor(to_avg)
        # Calculate mean across rows (dim=0)
        avg = t.mean(dim=0)
        # Rounding to 4 decimal places to match expected precision
        return [round(val, 4) for val in avg.tolist()]

    def concatenate(self, cat_one: list[list[float]], cat_two: list[list[float]]) -> list[list[float]]:
        t1 = torch.tensor(cat_one)
        t2 = torch.tensor(cat_two)
        # Concatenate side-by-side (dim=1)
        combined = torch.cat((t1, t2), dim=1)
        return combined.tolist()

    def get_loss(self, prediction: list[float], target: list[float]) -> float:
        pred = torch.tensor(prediction)
        targ = torch.tensor(target)
        # Built-in PyTorch functional MSE
        mse = F.mse_loss(pred, targ)
        return round(mse.item(), 4)