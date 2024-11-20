import torch
import torch.nn as nn

class StatelikeModule(nn.Module):
    def __init__(self, dt=1e-3):
        super().__init__()
        self.register_buffer('dt', torch.tensor(dt), persistent=False)
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
    def __init__(self, dt=1e-3, v_th=1.0, v_r=0.0):
        super().__init__(dt)
        self.register_buffer('v_th', torch.tensor(v_th), persistent=False)
        self.register_buffer('v_r', torch.tensor(v_r), persistent=False)
        self.learning_rules: list[LearningRule] = []

    def setup(self, x):
        super().setup(x)
        self.register_buffer('v', torch.zeros_like(x), persistent=False)
        self.register_buffer('t', torch.tensor(0, dtype=torch.int), persistent=False) # Timesteps elapsed since reset
        self.register_buffer('spikes', torch.zeros_like(x, dtype=torch.bool), persistent=False)
        self.register_buffer('spikes_accum', torch.zeros_like(x, dtype=torch.int32), persistent=False)
        for lr in self.learning_rules:
            self.add_module("learning_rule", lr)

    def forward(self, x):
        super().forward(x)
        self.t += 1 # Increment time
        for lr in self.learning_rules:
            lr.forward(x)
        return self.spikes
    
    def get_activation(self) -> torch.Tensor:
        # Assumed activation is proportional to the integrated current, excluding mapping error
        return self.spikes_accum * (self.v_th - self.v_r) / (self.t * self.dt)

class IF(SpikingNeuron):
    def __init__(self, dt=1e-3, v_th=1.0, v_r=0.0):
        super().__init__(dt, v_th, v_r)

    def forward(self, x):
        super().forward(x)
        self.v += x * self.dt
        self.spikes = self.v > self.v_th
        self.spikes_accum += self.spikes.int()
        self.v[self.spikes] -= (self.v_th - self.v_r) # Subtractive reset, more accurate
        return self.spikes



