import torch.utils
import torch.utils.data
from spieks.network.spiking_network import SpikingNetwork
import torch
import torch.nn as nn
from tqdm.notebook import tqdm

class Simulator():
	def __init__(
		self,
		net: SpikingNetwork, # The network to simulate.
		duration: float, # The duration of the experiment (in seconds).
		b_reset_net: bool = True # If true, call spieks.utils.reset_net on the net first.
	):
		self.net = net
		self.duration = duration
		self.b_reset_net = b_reset_net

		self.timesteps = int(duration / self.net.dt)

	def run(
		self,
		inputs: torch.Tensor,
		b_spiking_inputs: bool = True
	):
		assert inputs.dtype == torch.float

		if self.b_reset_net:
			self.net.reset()

		history_out = []
		with torch.no_grad():
			for i in range(self.timesteps):
				out = self.net(inputs[i] if b_spiking_inputs else inputs)
				history_out.append(out)
		out_mean = torch.stack(history_out).mean(dim=0) / self.net.dt
		return out_mean

class Classifier(Simulator):
	def __init__(
		self,
		net: SpikingNetwork, # The network to simulate.
		duration: float, # The duration of the experiment (in seconds).
		test_dataset: torch.utils.data.Dataset, # The testing dataset used to evaluate the net.
		b_reset_net: bool = True, # If true, call spieks.utils.reset_net on the net first.
		batch_size: int = 64, # The batch size to use while evaluating the net.
	):
		super().__init__(net, duration, b_reset_net)
		self.batch_size = batch_size
		self.dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)
		self.loss_fn = nn.CrossEntropyLoss()

	def run(
		self,
		b_spiking_inputs: bool = False, # Spiking inputs have size (B, T, N), non-spiking inputs have shape (B, N).
		device = None # The device to execute the classification.
	):
		# Use a forward hook to record outputs
		outputs = []
		def hook_record(module, input, output):
			outputs.append(input)
		list(self.net.children())[-1].register_forward_hook(hook_record)

		correct = 0
		total = 0
		running_loss = 0

		self.net.to(device)
		self.net.eval()
		for inputs, targets in tqdm(self.dataloader):
			inputs, targets = inputs.to(device), targets.to(device)
			total += len(targets)
			outputs = super().run(inputs, b_spiking_inputs)
			loss = self.loss_fn(outputs, targets)
			running_loss += loss * len(targets)
			pred = outputs.argmax(dim=1, keepdim=True) # get the index of the max log-probability
			correct += pred.eq(targets.view_as(pred)).sum().item()
		loss = running_loss / total
		accuracy = correct / total

		return loss, accuracy

