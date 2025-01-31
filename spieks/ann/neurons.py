import torch
import torch.nn as nn

# QCFS is the reproduction of the paper "QCFS"
class QCFS(nn.Module):
    def __init__(self, Q: int):
        super().__init__()
        self.Q = Q
        self.v_th = nn.Parameter(torch.tensor(float(Q)), requires_grad=True)
        self.p0 = torch.tensor(0.5)

    @staticmethod
    def floor_ste(x):
        return (x.floor() - x).detach() + x

    def extra_repr(self):
        return f"Q={self.Q}, p0={self.p0.item()}, v_th={self.v_th.item():.2f}"

    def forward(self, x):
        y = self.floor_ste(x * self.Q / self.v_th + self.p0)
        y = torch.clamp(y, 0, self.Q)
        return y * self.v_th / self.Q
