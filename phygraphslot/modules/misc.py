"""Miscellaneous modules."""

# FIXME

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import phygraphslot.lib.metrics as metrics
import phygraphslot.lib.metrics_jax as metrics_jax
import phygraphslot.modules.evaluator as evaluator
from phygraphslot.lib import utils
from phygraphslot.lib.utils import init_fn

DType = Any
Array = torch.Tensor
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]


class Readout(nn.Module):
	"""Module for reading out multiple targets from an embedding."""

	def __init__(self,
				 keys: Sequence[str],
				 readout_modules: nn.ModuleList,
				 stop_gradient: Optional[Sequence[bool]] = None
				):
		super().__init__()

		self.keys = keys
		self.readout_modules = readout_modules
		self.stop_gradient = stop_gradient

	def forward(self, inputs: Array) -> ArrayTree:
		num_targets = len(self.keys)
		assert num_targets >= 1, "Need to have at least one target."
		assert len(self.readout_modules) == num_targets, (
			f"len(modules):({len(self.readout_modules)}) and len(keys):({len(self.keys)}) must match.")
		if self.stop_gradient is not None:
			assert len(self.stop_gradient) == num_targets, (
			f"len(stop_gradient):({len(self.stop_gradient)}) and len(keys):({len(self.keys)}) must match.")
		outputs = {}
		modules_iter = iter(self.readout_modules)
		for i in range(num_targets):
			if self.stop_gradient is not None and self.stop_gradient[i]:
				x = x.detach() # FIXME
			else:
				x = inputs
			outputs[self.keys[i]] = next(modules_iter)(x)
		return outputs


class DummyReadout(nn.Module):

	def forward(self, inputs: Array) -> ArrayTree:
		return {}


class MLP(nn.Module):
	"""Simple MLP with one hidden layer and optional pre-/post-layernorm."""

	def __init__(self,
				 input_size: int,
				 hidden_size: int,
				 output_size: int, # if not given, should be inputs.shape[-1] at forward
				 num_hidden_layers: int = 1,
				 activation_fn: nn.Module = nn.ReLU,
				 layernorm: Optional[str] = None,
				 activate_output: bool = False,
				 residual: bool = False,
				 weight_init = None
				):
		super().__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_hidden_layers = num_hidden_layers
		self.activation_fn = activation_fn
		self.layernorm = layernorm
		self.activate_output = activate_output
		self.residual = residual
		self.weight_init = weight_init

		# submodules
		# layernorm
		if self.layernorm == "pre":
			self.layernorm_module = nn.LayerNorm(input_size, eps=1e-6)
		elif self.layernorm == "post":
			self.layernorm_module = nn.LayerNorm(output_size, eps=1e-6)
		# mlp
		self.model = nn.ModuleList()
		self.model.add_module("dense_mlp_0", nn.Linear(self.input_size, self.hidden_size))
		self.model.add_module("dense_mlp_0_act", self.activation_fn())
		for i in range(1, self.num_hidden_layers):
			self.model.add_module(f"den_mlp_{i}", nn.Linear(self.hidden_size, self.hidden_size))
			self.model.add_module(f"dense_mlp_{i}_act", self.activation_fn())
		self.model.add_module(f"dense_mlp_{self.num_hidden_layers}", nn.Linear(self.hidden_size, self.output_size))
		if self.activate_output:
			self.model.add_module(f"dense_mlp_{self.num_hidden_layers}_act", self.activation_fn())
		for name, module in self.model.named_children():
			if 'act' not in name:
				init_fn[weight_init['linear_w']](module.weight)
				init_fn[weight_init['linear_b']](module.bias)

	def forward(self, inputs: Array, train: bool = False) -> Array:
		del train # Unused

		x = inputs
		if self.layernorm == "pre":
			x = self.layernorm_module(x)
		for layer in self.model:
			x = layer(x)
		if self.residual:
			x = x + inputs
		if self.layernorm == "post":
			x = self.layernorm_module(x)
		return x


class myGRUCell(nn.Module):
	"""GRU cell as nn.Module

	Added because nn.GRUCell doesn't match up with jax's GRUCell...
	This one is designed to match ! (almost; output returns only once)

	The mathematical definition of the cell is as follows

  	.. math::

		\begin{array}{ll}
		r = \sigma(W_{ir} x + W_{hr} h + b_{hr}) \\
		z = \sigma(W_{iz} x + W_{hz} h + b_{hz}) \\
		n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
		h' = (1 - z) * n + z * h \\
		\end{array}
	"""

	def __init__(self,
				 input_size: int,
				 hidden_size: int,
				 gate_fn = torch.sigmoid,
				 activation_fn = torch.tanh,
				 weight_init = None
				):
		super().__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.gate_fn = gate_fn
		self.activation_fn = activation_fn
		self.weight_init = weight_init

		# submodules
		self.dense_ir = nn.Linear(input_size, hidden_size)
		self.dense_iz = nn.Linear(input_size, hidden_size)
		self.dense_in = nn.Linear(input_size, hidden_size)
		self.dense_hr = nn.Linear(hidden_size, hidden_size, bias=False)
		self.dense_hz = nn.Linear(hidden_size, hidden_size, bias=False)
		self.dense_hn = nn.Linear(hidden_size, hidden_size)
		self.reset_parameters()

	def reset_parameters(self) -> None:
		recurrent_weight_init = nn.init.orthogonal_
		if self.weight_init is not None:
			weight_init = init_fn[self.weight_init['linear_w']]
			bias_init = init_fn[self.weight_init['linear_b']]
		else:
			# weight init not given
			stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
			weight_init = bias_init = lambda weight: nn.init.uniform_(weight, -stdv, stdv)
		# input weights
		weight_init(self.dense_ir.weight)
		bias_init(self.dense_ir.bias)
		weight_init(self.dense_iz.weight)
		bias_init(self.dense_iz.bias)
		weight_init(self.dense_in.weight)
		bias_init(self.dense_in.bias)
		# hidden weights
		recurrent_weight_init(self.dense_hr.weight)
		recurrent_weight_init(self.dense_hz.weight)
		recurrent_weight_init(self.dense_hn.weight)
		bias_init(self.dense_hn.bias)
	
	def forward(self, inputs, carry):
		h = carry
		# input and recurrent layeres are summed so only one needs a bias
		r = self.gate_fn(self.dense_ir(inputs) + self.dense_hr(h))
		z = self.gate_fn(self.dense_iz(inputs) + self.dense_hz(h))
		# add bias because the linear transformations aren't directly summed
		n = self.activation_fn(self.dense_in(inputs) + r * self.dense_hn(h))
		new_h = (1. - z) * n + z * h
		return new_h


class PositionEmbedding(nn.Module):
	"""A module for applying N-dimensional position embedding.
	
	Attr:
		embedding_type: A string defining the type of position embedding to use.
			One of ["linear", "discrete_1d", "fourier", "gaussian_fourier"].
		update_type: A string defining how the input is updated with the position embedding.
			One of ["proj_add", "concat"].
		num_fourier_bases: The number of Fourier bases to use. For embedding_type == "fourier",
			the embedding dimensionality is 2 x number of position dimensions x num_fourier_bases. 
			For embedding_type == "gaussian_fourier", the embedding dimensionality is
			2 x num_fourier_bases. For embedding_type == "linear", this parameter is ignored.
		gaussian_sigma: Standard deviation of sampled Gaussians.
		pos_transform: Optional transform for the embedding.
		output_transform: Optional transform for the combined input and embedding.
		trainable_pos_embedding: Boolean flag for allowing gradients to flow into the position
			embedding, so that the optimizer can update it.
	"""

	def __init__(self,
				 input_shape: Tuple[int],
				 embedding_type: str,
				 update_type: str,
				 num_fourier_bases: int = 0,
				 gaussian_sigma: float = 1.0,
				 pos_transform: nn.Module = nn.Identity(),
				 output_transform: nn.Module = nn.Identity(),
				 trainable_pos_embedding: bool = False,
				 weight_init = None
				):
		super().__init__()

		self.input_shape = input_shape
		self.embedding_type = embedding_type
		self.update_type = update_type
		self.num_fourier_bases = num_fourier_bases
		self.gaussian_sigma = gaussian_sigma
		self.pos_transform = pos_transform
		self.output_transform = output_transform
		self.trainable_pos_embedding = trainable_pos_embedding
		self.weight_init = weight_init

		# submodules defined in module.
		self.pos_embedding = nn.Parameter(self._make_pos_embedding_tensor(input_shape),
										  requires_grad=self.trainable_pos_embedding)
		if self.update_type == "project_add":
			self.project_add_dense = nn.Linear(self.pos_embedding.shape[-1], input_shape[-1])
			# nn.init.xavier_uniform_(self.project_add_dense.weight)
			init_fn[weight_init['linear_w']](self.project_add_dense.weight)
			init_fn[weight_init['linear_b']](self.project_add_dense.bias)

	def _make_pos_embedding_tensor(self, input_shape):
		if self.embedding_type == "discrete_1d":
			# An integer tensor in [0, input_shape[-2]-1] reflecting
			# 1D discrete position encoding (encode the second-to-last axis).
			pos_embedding = np.broadcast_to(
				np.arange(input_shape[-2]), input_shape[1:-1])
		else:
			# A tensor grid in [-1, +1] for each input dimension.
			pos_embedding = utils.create_gradient_grid(input_shape[1:-1], [-1.0, 1.0])

		if self.embedding_type == "linear":
			pos_embedding = torch.from_numpy(pos_embedding)
		elif self.embedding_type == "discrete_1d":
			pos_embedding = F.one_hot(torch.from_numpy(pos_embedding), input_shape[-2])
		elif self.embedding_type == "fourier":
			# NeRF-style Fourier/sinusoidal position encoding.
			pos_embedding = utils.convert_to_fourier_features(
				pos_embedding * np.pi, basis_degree=self.num_fourier_bases)
			pos_embedding = torch.from_numpy(pos_embedding)
		elif self.embedding_type == "gaussian_fourier":
			# Gaussian Fourier features. Reference: https://arxiv.org/abs/2006.10739
			num_dims = pos_embedding.shape[-1]
			projection = np.random.normal(
				size=[num_dims, self.num_fourier_bases]) * self.gaussian_sigma
			pos_embedding = np.pi * pos_embedding.dot(projection)
			# A slightly faster implementation of sin and cos.
			pos_embedding = np.sin(
				np.concatenate([pos_embedding, pos_embedding + 0.5 * np.pi], axis=-1))
			pos_embedding = torch.from_numpy(pos_embedding)
		else:
			raise ValueError("Invalid embedding type provided.")
		
		# Add batch dimension.
		pos_embedding = pos_embedding.unsqueeze(0)
		pos_embedding = pos_embedding.float()

		return pos_embedding
	
	def forward(self, inputs: Array) -> Array:

		# Apply optional transformation on the position embedding.
		pos_embedding = self.pos_transform(self.pos_embedding).to(inputs.get_device())

		# Apply position encoding to inputs.
		if self.update_type == "project_add":
			# Here, we project the position encodings to the same dimensionality as
			# the inputs and add them to the inputs (broadcast along batch dimension).
			# This is roughly equivalent to concatenation of position encodings to the
			# inputs (if followed by a Dense layer), but is slightly more efficient.
			x = inputs + self.project_add_dense(pos_embedding)
		elif self.update_type == "concat":
			# Repeat the position embedding along the first (batch) dimension.
			pos_embedding = torch.broadcast_to(
				pos_embedding, inputs.shape[:-1] + pos_embedding.shape[-1:])
			# concatenate along the channel dimension.
			x = torch.concat((inputs, pos_embedding), dim=-1)
		else:
			raise ValueError("Invalid update type provided.")
		
		# Apply optional output transformation.
		x = self.output_transform(x)
		return x


#####################################################
# Losses

class ReconLoss(nn.Module):
	"""L2 loss."""
	
	def __init__(self, l2_weight=1, reduction="none"):
		super().__init__()

		self.l2 = nn.MSELoss(reduction=reduction)
		self.l2_weight = l2_weight

	def forward(self, model_outputs, batch):
		if isinstance(model_outputs, dict):
			pred_flow = model_outputs["outputs"]["flow"]
		else:
			pred_flow = model_outputs[1]
		video, boxes, segmentations, gt_flow, padding_mask, mask = batch

		# l2 loss between images and predicted images
		loss = self.l2_weight * self.l2(pred_flow, gt_flow)

		# sum over elements, leaving [B, -1]
		return loss.reshape(loss.shape[0], -1).sum(-1)


#######################################################
# Eval Metrics

class ARI(nn.Module):
	"""ARI."""

	def forward(self, model_outputs, batch, args):
		video, boxes, segmentations, flow, padding_mask, mask = batch

		pr_seg = model_outputs[0].squeeze(-1).int().cpu().numpy()
		gt_seg = segmentations.int().cpu().numpy()
		input_pad = padding_mask.cpu().numpy()
		mask = mask.cpu().numpy()

		ari_nobg = metrics_jax.Ari.from_model_output(
			predicted_segmentations=pr_seg, 
			ground_truth_segmentations=gt_seg,
			padding_mask=input_pad, 
			ground_truth_max_num_instances=args.max_instances + 1,
			predicted_max_num_instances=args.num_slots,
			ignore_background=True, mask=mask)
		
		return ari_nobg
