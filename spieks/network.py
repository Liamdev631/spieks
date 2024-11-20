from spieks.utils import reset_net
from spieks.neurons import SpikingNeuron
import torch
import torch.nn as nn

def run_sim(
    net: nn.Module, # The network to simulate.
    inputs: torch.Tensor, # The input data
    duration: float, # The duration of the experiment (in seconds).
    b_input_constant = True,
    b_reset_net: bool = True # If true, call spieks.utils.reset_net on the net first.
) : 
    assert inputs.dtype == torch.float

    # Determine the network DT from the first SpikingNeuron found. If the network
    # doesn't contain any, then we juts do a single pass and return.
    spiking_modules = [module for module in net.modules() if isinstance(module, SpikingNeuron)]
    DT = spiking_modules[0].dt if len(spiking_modules) > 0 else 0
    if DT == 0:
        return net(inputs) # Single ANN pass

    if b_reset_net:
        reset_net(net)

    trange = torch.arange(1, int(duration / DT) + 1) * DT
    if b_input_constant:
        for t in trange:
            net(inputs)
    else:
        for i, t in enumerate(trange):
            net(inputs[i])

    # Return the activation of the last layer
    return trange, spiking_modules[-1].get_activation()