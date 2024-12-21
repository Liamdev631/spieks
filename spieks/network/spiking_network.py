from spieks.neurons import StatelikeModule
import torch
import torch.nn as nn

def reset_module_recursive(net: nn.Module):
    """
    Recursively calls `reset()` on all modules in the network
    that are instances of `StatelikeModule`.
    
    Parameters:
    - net (nn.Module): The root PyTorch module.
    """
    # Call reset() if the module is an instance of StatelikeModule
    if isinstance(net, StatelikeModule):
        net.reset()

    # Recursively call reset_net for nested modules
    for module in net.children():
        reset_module_recursive(module)

class SpikingNetwork(nn.Module):
    def __init__(self,
        net: nn.Module,
        dt = 1e-3,
    ) -> None:
        super().__init__()
        self.net = net
        self.dt = dt

    def reset(self):
        reset_module_recursive(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward(x)
