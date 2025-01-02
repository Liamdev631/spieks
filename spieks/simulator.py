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
		if not isinstance(net, SpikingNetwork):
			raise TypeError("spieks.Simulator's are intended to be used with Spiking Networks only")

		self.net = net
		self.duration = duration
		self.b_reset_net = b_reset_net

		self.timesteps = int(duration / self.net.dt)

	def simulate(
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
				inp = inputs[i] if b_spiking_inputs else inputs
				out = self.net(inp)
				history_out.append(out)
		out_mean = torch.stack(history_out).mean(dim=0)
		return out_mean

class Classifier(Simulator):
	def __init__(
		self,
		net: SpikingNetwork, # The network to simulate.
		duration: float, # The duration of the experiment (in seconds).
		test_loader: torch.utils.data.DataLoader, # The testing dataset used to evaluate the net.
		b_reset_net: bool = True, # If true, call spieks.utils.reset_net on the net first.
	):
		super().__init__(net, duration, b_reset_net)
		self.test_loader = test_loader
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
		with torch.no_grad():
			for inputs, targets in tqdm(self.test_loader):
				inputs, targets = inputs.to(device), targets.to(device)
				total += len(targets)
				outputs = self.simulate(inputs, b_spiking_inputs)
				loss = self.loss_fn(outputs, targets)
				running_loss += loss * len(targets)
				pred = outputs.argmax(dim=1, keepdim=True) # get the index of the max log-probability
				correct += pred.eq(targets.view_as(pred)).sum().item()
		loss = running_loss / total
		accuracy = correct / total

		return loss, accuracy

