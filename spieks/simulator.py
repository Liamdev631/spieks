import torch.utils
import torch.utils.data
from spieks.network.spiking_network import SpikingNetwork
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import numpy as np

class Simulator():
	def __init__(
		self,
		net: SpikingNetwork, # The network to simulate.
		device = None, # The device to use when performing the simulation.
		b_reset_net: bool = True # If true, call spieks.utils.reset_net on the net first.
	):
		if not isinstance(net, SpikingNetwork):
			raise TypeError("spieks.Simulator's are intended to be used with Spiking Networks only")

		self.net = net
		self.device = device
		self.b_reset_net = b_reset_net

	def evaluate(
		self,
		input_data: torch.Tensor,
		duration: float = 1.0,
		b_spiking_inputs: bool = False,
	) -> torch.Tensor:
		
		timesteps = int(duration / self.net.dt)

		if self.b_reset_net:
			self.net.reset()

		output_tot = torch.zeros(timesteps).to(self.device)

		with torch.no_grad():
			for t in range(timesteps):
				inp = (input_data[t] if b_spiking_inputs else input_data)
				out = self.net(inp)
				output_tot += out
		activations = out / ((t + 1) * self.net.dt)
		return activations

	def evaulate_dataset(
		self,
		input_data: torch.utils.data.DataLoader,
		duration: float = 1.0,
		b_spiking_inputs: bool = False
	) -> torch.Tensor:

		outputs = []
		with torch.no_grad():
			for inputs, _ in tqdm(input_data):
				#targets = targets.to(self.device)
				out = self.evaluate(inputs, duration, b_spiking_inputs)
				outputs.append(out)
		out_history = torch.cat(out_history, dim=-2)
		return out_history

class Classifier(Simulator):
	def __init__(
		self,
		net: SpikingNetwork, # The network to simulate.
		device = None, # The device to use when performing the simulation.
		b_reset_net: bool = True, # If true, call spieks.utils.reset_net on the net first.
	):
		super().__init__(net, device, b_reset_net)
		self.loss_fn = nn.CrossEntropyLoss()

	def evaluate(
		self,
		input_data: torch.Tensor,
		target_data: torch.Tensor,
		duration: float = 1.0,
		b_spiking_inputs: bool = False
	) -> torch.Tensor:

		timesteps = int(duration / self.net.dt)

		self.net = self.net.to(self.device)
		input_data = input_data.to(self.device)
		target_data = target_data.to(self.device)

		if self.b_reset_net:
			self.net.reset()
		loss = torch.zeros(timesteps).to(self.device)
		correct = torch.zeros(timesteps).to(self.device)
		with torch.no_grad():
			output_tot = torch.zeros_like(self.net(input_data))
			for t in range(timesteps):
				input = (input_data[t] if b_spiking_inputs else input_data)
				out = self.net(input)
				output_tot += out

				# Calculate activation
				curr_time = (t + 1) * self.net.dt
				activations = output_tot / curr_time

				# Calculate loss
				loss[t] = self.loss_fn(activations, target_data)

				# Calculate accuracy
				correct[t] = (target_data == activations.max(1)[1]).mean(dtype=torch.float32)
		return activations, loss.detach().cpu().numpy(), correct.detach().cpu().numpy()

	def evaluate_dataset(
		self,
		dataloader: torch.utils.data.DataLoader,
		duration: float = 1.0,
		b_spiking_inputs: bool = False
	) -> torch.Tensor:

		all_activations = []
		all_loss = []
		all_accuracy = []

		with torch.no_grad():
			for inputs, targets in tqdm(dataloader):
				activations, loss, accuracy = self.evaluate(inputs, targets, duration, b_spiking_inputs)
				all_activations.append(activations)
				all_loss.append(loss)
				all_accuracy.append(accuracy)
		all_activations = torch.cat(all_activations, dim=-2)
		loss = np.mean(all_loss, axis=0)
		accuracy = np.mean(all_accuracy, axis=0)

		return all_activations, loss, accuracy
