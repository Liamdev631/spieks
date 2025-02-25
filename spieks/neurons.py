import torch
import torch.nn as nn
from math import sqrt

class StatelikeModule(nn.Module):
    def __init__(self, dt: float = 1e-3):
        super().__init__()
        self.dt = torch.tensor(dt, dtype=torch.float)
        self.initialized = False
    
    def setup(self, x: torch.Tensor):
        self.initialized = True

    def reset(self):
        self.initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.setup(x)
        return x
    
    def add_noise(self, x: torch.tensor, noise_std) -> None:
        if noise_std > 0:
            noise = torch.randn_like(x) * noise_std
            return x + noise
        return x
    
    def extra_repr(self):
        return f"log(dt)={torch.log10(self.dt):.0f}, v_r={self.v_r.data:.2f}, v_th={self.v_th.data:.2f}"

class LearningRule(StatelikeModule):
    ...

class SpikingNeuron(StatelikeModule):
    def __init__(self,
            dt: float = 1e-3,
            v_r: float = 0.0,
            v_th: float = 1.0,
    ):
        super().__init__(dt)
        self.v_r = torch.nn.Parameter(torch.tensor(v_r), requires_grad=False)
        self.v_th = torch.nn.Parameter(torch.tensor(v_th), requires_grad=False)
        self.learning_rules: list[LearningRule] = []

    def setup(self, x: torch.Tensor):
        super().setup(x)
        self.v = torch.zeros_like(x)
        self.spikes = torch.zeros_like(x, dtype=torch.bool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        super().forward(x)
        for lr in self.learning_rules:
            lr.forward(x)
        return self.spikes
    
    def extra_repr(self):
        return f"{super().extra_repr()}, log(dt)={torch.log10(self.dt):.2f}, clamp_v={self.v_clamp.data}"

class IF(SpikingNeuron):
    def __init__(self,
            dt: float = 1e-3,
            v_r: float = 0.0,
            v_th: float = 1.0,
            v_clamp: bool = False
    ):
        super().__init__(dt, v_r, v_th)
        self.rescale_factor = 1.0 / self.dt # Cancels dt scaling in integration
        self.v_clamp = torch.nn.Parameter(torch.tensor(v_clamp, dtype=torch.bool), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        super().forward(x)
        self.v += x * self.dt
        self.spikes = self.v > self.v_th
        self.v[self.spikes] -= (self.v_th - self.v_r) # Subtractive reset, more accurate
        if self.v_clamp:
            self.v.clamp_min_(self.v_r)
        return self.spikes.float() * (self.v_th - self.v_r) * self.rescale_factor

class NoisyIF(IF):
    def __init__(self,
        dt: float = 1e-3,
        v_r: float = 0.0,
        v_th: float = 1.0,
        v_clamp: bool = False,
        noise_std: float = 0.0
    ):
        super().__init__(dt, v_r, v_th, v_clamp)
        self.noise_std = torch.nn.Parameter(torch.tensor(noise_std / sqrt(dt)), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.add_noise(x, self.noise_std)
        return super().forward(x)

    def extra_repr(self):
        return f"{super().extra_repr()}, noise_std={self.noise_std.data:.3f}"
