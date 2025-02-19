import torch
import torch.nn as nn

class GradFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

qcfsFloor = GradFloor.apply

# QCFS is the reproduction of the paper "QCFS"
class QCFS(nn.Module):
    def __init__(self, Q: int = 8):
        super().__init__()
        self.Q = Q
        self.v_th = nn.Parameter(torch.tensor(float(Q)), requires_grad=True)
        self.p0 = torch.tensor(0.5)

    @staticmethod
    def floor_ste(x):
        return (x.floor() - x).detach() + x

    def extra_repr(self):
        return f"Q={self.Q}, p0={self.p0.item():.2f}, v_th={self.v_th.item():.2f}"

    def forward(self, x):
        # y = self.floor_ste(x * self.Q / self.v_th + self.p0)
        # y = torch.clamp(y, 0, self.Q)
        # return y * self.v_th / self.Q
        x = x / self.v_th
        x = torch.clamp(x, 0, 1)
        x = qcfsFloor(x * self.Q + self.p0) / self.Q
        x = x * self.v_th
        return x
