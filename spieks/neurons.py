import torch
import torch.nn as nn

class StatelikeModule(nn.Module):
    def __init__(self, dt=1e-3):
        super().__init__()
        self.dt = torch.tensor(dt)
        self.initialized = False
    
    def setup(self, x):
        self.initialized = True

    def reset(self):
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            self.setup(x)
        return x

class LearningRule(StatelikeModule):
    ...

class SpikingNeuron(StatelikeModule):
    def __init__(self, dt=1e-3, v_r=0.0, v_th=1.0):
        super().__init__(dt)
        self.v_r = v_r
        self.v_th = v_th
        self.learning_rules: list[LearningRule] = []

    def setup(self, x):
        super().setup(x)
        self.v = torch.zeros_like(x)
        self.spikes = torch.zeros_like(x, dtype=torch.bool)

    def forward(self, x):
        super().forward(x)
        for lr in self.learning_rules:
            lr.forward(x)
        return self.spikes

class IF(SpikingNeuron):
    def __init__(self, dt=1e-3, v_r=0.0, v_th=1.0):
        super().__init__(dt, v_r, v_th)
        self.rescale_factor = 1.0 / self.dt # Cancels dt scaling in integration

    def forward(self, x):
        super().forward(x)
        self.v += x * self.dt
        self.spikes = self.v > self.v_th
        self.v[self.spikes] -= (self.v_th - self.v_r) # Subtractive reset, more accurate
        return self.spikes.float() * (self.v_th - self.v_r) * self.rescale_factor

    def extra_repr(self):
        return f"log(dt)={torch.log10(self.dt):.0f}, v_r={self.v_r}, v_th={self.v_th}"
