import torch
import torch.nn as nn
from spieks.neurons import StatelikeModule, SpikingNeuron

class LearningRule(StatelikeModule):
    def __init__(self, parent: SpikingNeuron, tau=1e-1):
        super().__init__()
        self.parent = parent
        if self not in self.parent.learning_rules:
            self.parent.learning_rules.append(self)
        self.register_buffer('tau', torch.tensor(tau), persistent=False)

class SpikeTraceLearningRule(LearningRule):
    def __init__(self, parent: SpikingNeuron):
        super().__init__(parent)
    
    def setup(self, x):
        super().setup(x)
        self.register_buffer('spike_trace', torch.zeros_like(x), persistent=False)

    def forward(self, x):
        super().forward(x)
        self.spike_trace[self.parent.spikes] = 1.0
        self.spike_trace *= (1 - self.dt * self.tau)
        return self.spike_trace

