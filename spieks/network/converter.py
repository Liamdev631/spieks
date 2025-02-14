from spieks.network import SpikingNetwork
from spieks.neurons import SpikingNeuron, IF
from torch.__future__ import set_overwrite_module_params_on_conversion
import torch.nn as nn
import copy

UNCONVERTABLE_MODULES = [
	nn.MaxPool1d,
	nn.MaxPool2d,
	nn.MaxPool3d,
	nn.MaxUnpool1d,
	nn.MaxUnpool2d,
	nn.MaxUnpool3d,
	nn.AdaptiveMaxPool1d,
	nn.AdaptiveMaxPool2d,
	nn.AdaptiveMaxPool3d,
	nn.FractionalMaxPool2d,
	nn.FractionalMaxPool3d,
]

class Converter():
	@staticmethod
	def convert(
		model: nn.Module,
		model_subs: dict[nn.Module, SpikingNeuron] = { nn.ReLU: IF },
		neuron_args: dict = {}
	) -> SpikingNetwork:
		
		# Deepcopy the original ANN
		set_overwrite_module_params_on_conversion(True)
		new_net = copy.deepcopy(model)

		# Replace all layers in the network according to 'replacements'
		for (old_layer_type, new_layer_type) in model_subs.items():
			new_net = swap_layers(new_net, old_layer_type, new_layer_type, neuron_args)

		# Perform a final validation check on the model
		for name, module in new_net.named_children():
			if type(module) in UNCONVERTABLE_MODULES:
				raise TypeError(f"Module {name} of type {type(module)} cannot be used in an SNN!")

		return SpikingNetwork(new_net, neuron_args["dt"])

def swap_layers(parent: nn.Module, old_layer_type: type[nn.Module], new_layer_type: type[nn.Module], neuron_args={}):
	for name, module in parent.named_children():
		if isinstance(module, old_layer_type):

			if old_layer_type in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
				raise ValueError(f"Cannot convert {old_layer_type} to {new_layer_type}")

			# Extract relevant parameters specific certain layers
			if old_layer_type == nn.MaxPool2d and new_layer_type == nn.AvgPool2d:
				neuron_args['kernel_size'] = module.kernel_size
				neuron_args['stride'] = module.stride
				neuron_args['padding'] = module.padding
				neuron_args['ceil_mode'] = module.ceil_mode
				neuron_args['count_include_pad'] = getattr(module, 'count_include_pad', True)

			module.state_dict(destination=neuron_args)

			# Ensure neuron_args are passed correctly
			setattr(parent, name, new_layer_type(**neuron_args))
		elif isinstance(module, nn.Module):
			swap_layers(module, old_layer_type, new_layer_type, neuron_args)
	return parent
