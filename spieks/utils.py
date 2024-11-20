from spieks.neurons import StatelikeModule
import torch.nn as nn

def reset_net(net: nn.Module):
    for module in net:
        if isinstance(module, StatelikeModule):
            module.reset()
