# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from pyparsing import line_end
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import dataclasses

import torch
import torch.nn as nn
import torch.distributed as dist
from torch._six import inf

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb

import wandb

class SmoothedValue(object):
	"""Track a series of values and provide access to smoothed values over a
	window or the global series average.
	"""

	def __init__(self, window_size=20, fmt=None):
		if fmt is None:
			fmt = "{median:.4f} ({global_avg:.4f})"
		self.deque = deque(maxlen=window_size)
		self.total = 0.0
		self.count = 0
		self.fmt = fmt

	def update(self, value, n=1):
		self.deque.append(value)
		self.count += n
		self.total += value * n

	def synchronize_between_processes(self):
		"""
		Warning: does not synchronize the deque!
		"""
		if not is_dist_avail_and_initialized():
			return
		t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
		dist.barrier()
		dist.all_reduce(t)
		t = t.tolist()
		self.count = int(t[0])
		self.total = t[1]

	@property
	def median(self):
		d = torch.tensor(list(self.deque))
		return d.median().item()

	@property
	def avg(self):
		d = torch.tensor(list(self.deque), dtype=torch.float32)
		return d.mean().item()

	@property
	def global_avg(self):
		return self.total / self.count

	@property
	def max(self):
		return max(self.deque)

	@property
	def value(self):
		return self.deque[-1]

	def __str__(self):
		return self.fmt.format(
			median=self.median,
			avg=self.avg,
			global_avg=self.global_avg,
			max=self.max,
			value=self.value)


class MetricLogger(object):
	def __init__(self, delimiter="\t"):
		self.meters = defaultdict(SmoothedValue)
		self.delimiter = delimiter

	def update(self, **kwargs):
		for k, v in kwargs.items():
			if v is None:
				continue
			if isinstance(v, torch.Tensor):
				v = v.item()
			assert isinstance(v, (float, int))
			self.meters[k].update(v)

	def __getattr__(self, attr):
		if attr in self.meters:
			return self.meters[attr]
		if attr in self.__dict__:
			return self.__dict__[attr]
		raise AttributeError("'{}' object has no attribute '{}'".format(
			type(self).__name__, attr))

	def __str__(self):
		loss_str = []
		for name, meter in self.meters.items():
			loss_str.append(
				"{}: {}".format(name, str(meter))
			)
		return self.delimiter.join(loss_str)

	def synchronize_between_processes(self):
		for meter in self.meters.values():
			meter.synchronize_between_processes()

	def add_meter(self, name, meter):
		self.meters[name] = meter

	def log_every(self, iterable, print_freq, header=None):
		i = 0
		if not header:
			header = ''
		start_time = time.time()
		end = time.time()
		iter_time = SmoothedValue(fmt='{avg:.4f}')
		data_time = SmoothedValue(fmt='{avg:.4f}')
		space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
		log_msg = [
			header,
			'[{0' + space_fmt + '}/{1}]',
			'eta: {eta}',
			'{meters}',
			'time: {time}',
			'data: {data}'
		]
		if torch.cuda.is_available():
			log_msg.append('max mem: {memory:.0f}')
		log_msg = self.delimiter.join(log_msg)
		MB = 1024.0 * 1024.0
		for obj in iterable:
			data_time.update(time.time() - end)
			yield obj
			iter_time.update(time.time() - end)
			if i % print_freq == 0 or i == len(iterable) - 1:
				eta_seconds = iter_time.global_avg * (len(iterable) - i)
				eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
				if torch.cuda.is_available():
					print(log_msg.format(
						i, len(iterable), eta=eta_string,
						meters=str(self),
						time=str(iter_time), data=str(data_time),
						memory=torch.cuda.max_memory_allocated() / MB))
				else:
					print(log_msg.format(
						i, len(iterable), eta=eta_string,
						meters=str(self),
						time=str(iter_time), data=str(data_time)))
			i += 1
			end = time.time()
		total_time = time.time() - start_time
		total_time_str = str(datetime.timedelta(seconds=int(total_time)))
		print('{} Total time: {} ({:.4f} s / it)'.format(
			header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
	"""
	This function disables printing when not in master process
	"""
	builtin_print = builtins.print

	def print(*args, **kwargs):
		force = kwargs.pop('force', False)
		force = force or (get_world_size() > 8)
		if is_master or force:
			now = datetime.datetime.now().time()
			builtin_print('[{}] '.format(now), end='')  # print with time stamp
			builtin_print(*args, **kwargs)

	builtins.print = print


def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True


def get_world_size():
	if not is_dist_avail_and_initialized():
		return 1
	return dist.get_world_size()


def get_rank():
	if not is_dist_avail_and_initialized():
		return 0
	return dist.get_rank()


def is_main_process():
	return get_rank() == 0


def save_on_master(*args, **kwargs):
	if is_main_process():
		torch.save(*args, **kwargs)


def init_distributed_mode(args):
	if args.dist_on_itp:
		args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
		args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
		args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
		args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
		os.environ['LOCAL_RANK'] = str(args.gpu)
		os.environ['RANK'] = str(args.rank)
		os.environ['WORLD_SIZE'] = str(args.world_size)
		# ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
	elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
		args.rank = int(os.environ["RANK"])
		args.world_size = int(os.environ['WORLD_SIZE'])
		args.gpu = int(os.environ['LOCAL_RANK'])
	elif 'SLURM_PROCID' in os.environ:
		args.rank = int(os.environ['SLURM_PROCID'])
		args.gpu = args.rank % torch.cuda.device_count()
	else:
		print('Not using distributed mode')
		setup_for_distributed(is_master=True)  # hack
		args.distributed = False
		return

	args.distributed = True

	torch.cuda.set_device(args.gpu)
	args.dist_backend = 'nccl'
	print('| distributed init (rank {}): {}, gpu {}'.format(
		args.rank, args.dist_url, args.gpu), flush=True)
	torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
										 world_size=args.world_size, rank=args.rank)
	torch.distributed.barrier()
	setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
	state_dict_key = "amp_scaler"

	def __init__(self):
		self._scaler = torch.cuda.amp.GradScaler()

	def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
		self._scaler.scale(loss).backward(create_graph=create_graph)
		if update_grad:
			if clip_grad is not None:
				assert parameters is not None
				self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
				norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
			else:
				self._scaler.unscale_(optimizer)
				norm = get_grad_norm_(parameters)
			self._scaler.step(optimizer)
			self._scaler.update()
		else:
			norm = None
		return norm

	def state_dict(self):
		return self._scaler.state_dict()

	def load_state_dict(self, state_dict):
		self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
	if isinstance(parameters, torch.Tensor):
		parameters = [parameters]
	parameters = [p for p in parameters if p.grad is not None]
	norm_type = float(norm_type)
	if len(parameters) == 0:
		return torch.tensor(0.)
	device = parameters[0].grad.device
	if norm_type == inf:
		total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
	else:
		total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
	return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
	output_dir = Path(args.output_dir)
	epoch_name = str(epoch)
	if loss_scaler is not None:
		checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
		for checkpoint_path in checkpoint_paths:
			to_save = {
				'model': model_without_ddp.state_dict(),
				'optimizer': optimizer.state_dict(),
				'epoch': epoch,
				'scaler': loss_scaler.state_dict(),
				'args': args,
			}

			save_on_master(to_save, checkpoint_path)
	else:
		client_state = {'epoch': epoch}
		model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
	if args.resume:
		if args.resume.startswith('https'):
			checkpoint = torch.hub.load_state_dict_from_url(
				args.resume, map_location='cpu', check_hash=True)
		else:
			checkpoint = torch.load(args.resume, map_location='cpu')
		model_without_ddp.load_state_dict(checkpoint['model'])
		print("Resume checkpoint %s" % args.resume)
		if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
			optimizer.load_state_dict(checkpoint['optimizer'])
			args.start_epoch = checkpoint['epoch'] + 1
			if 'scaler' in checkpoint:
				loss_scaler.load_state_dict(checkpoint['scaler'])
			print("With optim & sched!")


def all_reduce_mean(x):
	world_size = get_world_size()
	if world_size > 1:
		x_reduce = torch.tensor(x).cuda()
		dist.all_reduce(x_reduce)
		x_reduce /= world_size
		return x_reduce.item()
	else:
		return x


####################################
# I added these

@dataclasses.dataclass
class ParamRow:
  name: str
  shape: Tuple[int]
  size: int


@dataclasses.dataclass
class ParamRowWithStats(ParamRow):
  mean: float
  std: float

def _default_table_value_formatter(value):
  """Formats ints with "," between thousands and floats to 3 digits."""
  if isinstance(value, bool):
    return str(value)
  elif isinstance(value, int):
    return "{:,}".format(value)
  elif isinstance(value, float):
    return "{:.3}".format(value)
  else:
    return str(value)

def make_table(
    rows: List[Any],
    *,
    column_names: Optional[Sequence[str]] = None,
    value_formatter: Callable[[Any], str] = _default_table_value_formatter,
    max_lines: Optional[int] = None,
) -> str:
  """Renders a list of rows to a table.

  Args:
    rows: List of dataclass instances of a single type (e.g. `ParamRow`).
    column_names: List of columns that that should be included in the output. If
      not provided, then the columns are taken from keys of the first row.
    value_formatter: Callable used to format cell values.
    max_lines: Don't render a table longer than this.

  Returns:
    A string representation of the table in the form:

    +---------+---------+
    | Col1    | Col2    |
    +---------+---------+
    | value11 | value12 |
    | value21 | value22 |
    +---------+---------+
  """

  if any(not dataclasses.is_dataclass(row) for row in rows):
    raise ValueError("Expected `rows` to be list of dataclasses")
  if len(set(map(type, rows))) > 1:
    raise ValueError("Expected elements of `rows` be of same type.")

  class Column:

    def __init__(self, name, values):
      self.name = name.capitalize()
      self.values = values
      self.width = max(len(v) for v in values + [name])

  if column_names is None:
    if not rows:
      return "(empty table)"
    column_names = [field.name for field in dataclasses.fields(rows[0])]

  columns = [
      Column(name, [value_formatter(getattr(row, name))
                    for row in rows])
      for name in column_names
  ]

  var_line_format = "|" + "".join(f" {{: <{c.width}s}} |" for c in columns)
  sep_line_format = var_line_format.replace(" ", "-").replace("|", "+")
  header = var_line_format.replace(">", "<").format(*[c.name for c in columns])
  separator = sep_line_format.format(*["" for c in columns])

  lines = [separator, header, separator]
  for i in range(len(rows)):
    if max_lines and len(lines) >= max_lines - 3:
      lines.append("[...]")
      break
    lines.append(var_line_format.format(*[c.values[i] for c in columns]))
  lines.append(separator)

  return "\n".join(lines)

def parameter_overview(model: nn.Module):
	rows = []
	for name, value in model.named_parameters():
		rows.append(ParamRowWithStats(
			name=name, shape=tuple(value.shape),
			size=int(np.prod(value.shape)),
			mean=float(value.mean()),
			std=float(value.std())))
	total_weights = sum([np.prod(v.shape) for v in model.parameters()])
	column_names = [field.name for field in dataclasses.fields(ParamRowWithStats)]
	table = make_table(rows, column_names=column_names)
	return table + f"\nTotal: {total_weights:,}"

# TODO: make output path absolute and not assuming an experiments dir
def save_snapshot(args, model, optimizer, global_step, output_fn):
	print('saving model.')
	os.makedirs(os.path.dirname(output_fn), exist_ok=True)
	payload = {
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'global_step': global_step,
		'args': args
	}
	torch.save(payload, output_fn)
	print('saved model.')


def load_snapshot(model, optimizer, device, name):
	print('loading model.')
	snapshot_path = name
	payload = torch.load(snapshot_path, map_location=device)
	model.load_state_dict(payload['model'])
	optimizer.load_state_dict(payload['optimizer'])
	print('loaded model.')
	return payload['args'], payload['global_step']

#######################

def plot_image(ax, img, label=None):
		ax.imshow(img)
		ax.axis('off')
		ax.set_xticks([])
		ax.set_yticks([])
		if label:
			# ax.set_title(label, fontsize=3, y=-21)
			ax.set_xlabel(label, fontsize=3)
			ax.axis('on')


def visualize(vid,
			gt_mask,
			pr_mask,
			gt_flow,
			pr_frame,
			pr_flow,
			mask,
			args,
			name,
			send_to_wandb=False):
	"""
	Plot the video, gt seg and pred seg masks

	Args:
		vid: (L H W C)
		gt_mask: (L H W C)
		pr_mask: (L H W C)
		gt_flow: undo
		pr_frame: (L H W C)
		pr_flow: (L H W C)
		mask: (L num_objects H W)
		args: args
		send_to_wandb: True or False
	"""

	trunk = 10
	output_fn = f"./experiments/{args.group}_{args.exp}/viz_seg/{name}.png"
	T = min(len(vid), trunk)
	os.makedirs(os.path.dirname(output_fn), exist_ok=True)

	plt.close()
	fig, ax = plt.subplots(T, 3, dpi=400)

	for t in range(T):
		gt_seg = label2rgb(gt_mask[t], vid[t])
		pred_seg = label2rgb(pr_mask[t], vid[t])

		plot_image(ax[t, 0], vid[t], 'original')
		plot_image(ax[t, 1], gt_seg, 'gt_seg')
		plot_image(ax[t, 2], pred_seg, 'pred_seg')

	plt.savefig(output_fn)
	# plt.show()
	# time.sleep(1)
	# plt.close()

	if send_to_wandb:
		wandb.log({
			"eval/seg":
			wandb.Image(plt.gcf())
		})


	trunk = 8
	output_fn = f"./experiments/{args.group}_{args.exp}/viz_slots_flow/{name}.png"
	T = min(len(vid), trunk)
	n_objs = mask.shape[1]
	os.makedirs(os.path.dirname(output_fn), exist_ok=True)

	slots = vid[:, np.newaxis, :, :, :] * mask[:, :, :, :, np.newaxis]

	plt.close()
	fig, ax = plt.subplots(T, n_objs+3, dpi=400)

	for t in range(T):
		
		plot_image(ax[t, 0], vid[t], 'frame')
		plot_image(ax[t, 1], gt_flow[t], 'gt_flow')
		plot_image(ax[t, 2], pr_flow[t], 'pred_flow')

		for obj in range(3, n_objs+3):
			plot_image(ax[t, obj], slots[t, obj-3], f'slot {obj-2}')
	
	plt.savefig(output_fn)
	# plt.show()
	# time.sleep(1)
	# plt.close()

	if send_to_wandb:
		wandb.log({
			"eval/slots_flow":
			wandb.Image(plt.gcf())
		})

	if args.model_type == "flow":
		trunk = 6
		output_fn = f"./experiments/{args.group}_{args.exp}/viz_slots_frame_pred/{name}.png"
		T = min(len(vid), trunk)
		n_objs = mask.shape[1]
		os.makedirs(os.path.dirname(output_fn), exist_ok=True)

		slots = vid[:, np.newaxis, :, :, :] * mask[:, :, :, :, np.newaxis]

		plt.close()
		fig, ax = plt.subplots(T, n_objs+3, dpi=400)

		for t in range(T):

			plot_image(ax[t, 0], vid[t], 'frame')
			plot_image(ax[t, 1], pr_frame[t], 'pred_frame')
			plot_image(ax[t, 2], pr_flow[t], 'pred_flow')

			for obj in range(3, n_objs+3):
				plot_image(ax[t, obj], slots[t, obj-3], f'slot {obj-2}')

		plt.savefig(output_fn)
		# plt.show()
		# time.sleep(1)
		# plt.close()

		if send_to_wandb:
			wandb.log({
				"eval/slots_pred_frame":
				wandb.Image(plt.gcf())
			})
