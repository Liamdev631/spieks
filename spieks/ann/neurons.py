import torch
import torch.nn as nn

# QCFS is the reproduction of the paper "QCFS"
class QCFS(nn.Module):
    def __init__(self, Q: int):
        super().__init__()
        self.Q = Q

    @staticmethod
    def floor_ste(x):
        return (x.floor() - x).detach() + x

    def extra_repr(self):
        return f"Q={self.Q}"

    def forward(self, x):
        y = self.floor_ste(x * self.Q + 0.5)
        y = torch.clamp(y, 0, self.Q)
        return y / self.Q
