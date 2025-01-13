from spieks.network import SpikingNetwork
from spieks.neurons import SpikingNeuron, IF
from torch.__future__ import set_overwrite_module_params_on_conversion
import torch.nn as nn
import copy

class Converter():
    @staticmethod
    def convert(
        model: nn.Module,
        dt: float = 1e-3,
        model_subs: dict[nn.Module, SpikingNeuron] = { nn.ReLU: IF },
        neuron_args: dict = {}
    ) -> SpikingNetwork:
        # Deepcopy the original ANN
        set_overwrite_module_params_on_conversion(True)
        new_net = copy.deepcopy(model)

        neuron_args.update({'dt': dt})

        # Replace all layers in the network according to 'replacements'
        for (old_layer_type, new_layer_type) in model_subs.items():
            new_net = swap_layers(new_net, old_layer_type, new_layer_type, neuron_args)

        return SpikingNetwork(new_net, dt)

def swap_layers(model: nn.Module, old_layer_type: type[nn.Module], new_layer_type: type[nn.Module], neuron_args={}):
    for name, module in model.named_children():
        if isinstance(module, old_layer_type):
            setattr(model, name, new_layer_type(*module.parameters(), **neuron_args))
        elif isinstance(module, nn.Module):
            swap_layers(module, old_layer_type, new_layer_type, neuron_args)
    return model
