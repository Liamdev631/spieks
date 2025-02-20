import torch
import torch.nn as nn
from math import sqrt

class StatelikeModule(nn.Module):
    def __init__(self, dt=1e-3):
        super().__init__()
        self.dt = torch.tensor(dt)
        self.initialized = False
    
    def setup(self, x):
        self.initialized = True

    def reset(self):
        self.initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.setup(x)
        return x
    
    def add_noise(self, x, noise_std) -> None:
        if noise_std > 0:
            noise = torch.randn_like(x) * noise_std
            return x + noise
        return x
    
    def extra_repr(self):
        return f"log(dt)={torch.log10(self.dt):.0f}"

class LearningRule(StatelikeModule):
    ...

class SpikingNeuron(StatelikeModule):
    def __init__(self, dt=1e-3, v_r=0.0, v_th=1.0):
        super().__init__(dt)
        self.v_r = torch.nn.Parameter(torch.tensor(v_r), requires_grad=False)
        self.v_th = torch.nn.Parameter(torch.tensor(v_th), requires_grad=False)
        self.learning_rules: list[LearningRule] = []

    def setup(self, x):
        super().setup(x)
        self.v = torch.zeros_like(x)
        self.spikes = torch.zeros_like(x, dtype=torch.bool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        super().forward(x)
        for lr in self.learning_rules:
            lr.forward(x)
        return self.spikes
    
    def extra_repr(self):
        return f"log(dt)={torch.log10(self.dt):.2f}, v_r={self.v_r:.2f}, v_th={self.v_th:.2f}"

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

class NoisyIF(IF):
    def __init__(self, dt=1e-3, v_r=0.0, v_th=1.0, noise_std=0.0):
        super().__init__(dt, v_r, v_th)
        self.noise_std = torch.nn.Parameter(torch.tensor(noise_std / sqrt(dt)), required_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.add_noise(x, self.noise_std)
        return super().forward(x)

    def extra_repr(self):
        return f"log(dt)={torch.log10(self.dt):.0f}, v_r={self.v_r}, v_th={self.v_th}, noise_std={self.noise_std}"
