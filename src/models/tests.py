import torch
import numpy as np
import numpy.matlib
from torch.utils.data import DataLoader
from math import sqrt, pi, cosh
from datetime import datetime

import logging

from ..dataloaders import collate_fn

def permutation_test(model, data):
	logging.info('Beginning permutation test!')

	mask = data['particle_mask']

	# Generate a list of indices for each molecule.
	# We will generate a permutation only for the particles that exist (are not masked.)
	batch_size, particle_size = mask.shape
	perm = 1*torch.arange(particle_size).expand(batch_size, -1)
	for idx in range(batch_size):
		num_particles = (mask[idx, :].long()).sum()
		perm[idx, :num_particles] = torch.randperm(num_particles)

	apply_perm = lambda dim, mat: torch.stack([torch.index_select(mat[idx], dim, p) for (idx, p) in enumerate(perm)])

	assert((mask == apply_perm(0, mask)).all())

	data_noperm = data
	data_perm = {key: apply_perm(0, val) if key in ['Pmu', 'scalars'] else val for key, val in data.items()}

	outputs_noperm = model(data_noperm)['predict']
	outputs_perm = model(data_perm)['predict']

	invariance_test = (outputs_perm - outputs_noperm).abs().max()/outputs_noperm.abs().max()

	logging.info('Permutation Invariance test error: {}'.format(invariance_test))
	

def batch_test(model, data):
	logging.info('Beginning batch invariance test!')
	data_split = {key: val.unbind(dim=0) if (torch.is_tensor(val) and val.numel() > 1) else val for key, val in data.items()}
	data_split = [{key: val[idx].unsqueeze(0) if type(val) is tuple else val for key, val in data_split.items()} for idx in range(len(data['is_signal']))]

	outputs_split = torch.cat([model(data_sub) for data_sub in data_split])
	outputs_full = model(data)
	invariance_test = (outputs_split - outputs_full).abs().max()/outputs_full.abs().mean()

	logging.info('Batch invariance test error: {}'.format(invariance_test))


def ir_data(data_irc, num_particles=1, alpha=0):
	"""
	Here we replace the last num_particles empty inputs with random 4-momenta of scale alpha (zero if alpha=0).
	Under IR-safety, injecting a zero-momentum particle should leave the output unchanged.
	If alpha=0, the entire change amounts to simply switching the last num_particles entries of data['particle_mask'] to True.
	If alpha!=0, we generate random 3-momenta and then define energies as |p| + alpha*torch.rand, so that the resulting particles are timelike.
	An IR-safe network's output should remain unchanged when alpha=0 and only slightly perturbed for small nonzero alpha.
	This can be verified by running PELICAN with the --ir-safe flag.
	"""

	batch_size = data_irc['Nobj'].shape[0]
	rand_pmus3 = alpha * (2 * torch.rand(batch_size, num_particles, 3, dtype = data_irc['Pmu'].dtype, device = data_irc['Pmu'].device) - 1)
	rand_pmus = torch.cat([rand_pmus3.norm(dim=2, keepdim=True) + alpha * torch.rand(batch_size, num_particles, 1,  dtype = data_irc['Pmu'].dtype, device = data_irc['Pmu'].device), rand_pmus3], dim=2)
	data_irc['Pmu'][:, -num_particles:] = rand_pmus
	idx = 4 if data_irc['Pmu'].shape[1] > 4 else 2
	if 'scalars' in data_irc.keys():
		data_irc['scalars'][:, -num_particles:] = data_irc['scalars'][:, -num_particles:] + data_irc['scalars'][:, [idx]]
	data_irc['particle_mask'][:, -num_particles:] = True
	data_irc['edge_mask'] = data_irc['particle_mask'].unsqueeze(1) * data_irc['particle_mask'].unsqueeze(2)
	data_irc['Nobj'] = data_irc['Nobj'] + num_particles
	return data_irc

def c_data(data_irc):
	"""
	Take two (or more) massless collinear input particles, p1 and p2, replace them with two particles with momenta (p1+p2) and 0, compare outputs.
	Assuming the batch was prepared using expand_data, this means the two copies of [1,0,0,1] get replaced by [[1.5,0,0,1.5],[0.5,0,0,0.5]]. 
	A C-safe network must produce the same output in both cases. This can be enforced by the --c-safe flag.
	"""
	
	# Which COLLINEAR MASSLESS input particles to use. We have deliberately inserted two particles with p=[1,0,0,1] in irc_test() so that indices=[0,1] can be used
	indices = [0,1]  

	# Half of the total momentum of all the particles except the first gets shifted onto the first
	total_momentum = data_irc['Pmu'][:,indices[1:],:].sum(dim=1)
	data_irc['Pmu'][:,indices[1:],:] = data_irc['Pmu'][:,indices[1:],:] / 2
	data_irc['Pmu'][:,indices[0],:] = data_irc['Pmu'][:,indices[0],:] + total_momentum / 2
	return data_irc

def irc_data(data_irc):
	"""
	Does an irc-split: creates one empty particle slot with particle_mask=True (which is an IR-split) and then 
	moves half of the momentum stored in the first particle slot (usually the beam [1,0,0,1]) into that slot.
	This should lead to the same outputs in an IRC-safe network.
	This transformation is used for real data in place of c_data because c_data requires two pre-existing 
	collinear massless particles, which can't be expected of real data.
	"""
	# Which MASSLESS input particles to use. Normally index=0, one of the beams.
	index = 0

	data_irc['particle_mask'][:, -1] = True
	data_irc['edge_mask'] = data_irc['particle_mask'].unsqueeze(1) * data_irc['particle_mask'].unsqueeze(2)
	data_irc['Nobj'] = data_irc['Nobj'] + 1

	total_momentum = data_irc['Pmu'][:,index]
	data_irc['Pmu'][:,-1] = total_momentum/2
	data_irc['Pmu'][:,index]  = total_momentum/2
	return data_irc

def expand_data(data, num_particles=1, c=False):
	"""
	Prepares a batch for the IR/C tests.
	Inserts two copies of the 4-vector [1,0,0,1] to the beginning of each event with particle_mask=True.
	Further inserts num_particles copies of [0,0,0,0] with particle_mask=False.
	"""

	batch_size = data['Nobj'].shape[0]
	zero = torch.tensor(0.)

	zero_pmus = torch.zeros(batch_size, num_particles, 4, dtype = data['Pmu'].dtype, device = data['Pmu'].device)
	if c:
		beam = torch.tensor([[[1,0,0,1],[1,0,0,1]]], dtype=data['Pmu'].dtype).expand(data['Pmu'].shape[0], 2, 4)
		data['Pmu'] = torch.cat([beam, data['Pmu']], 1)
	# data['Pmu'] = torch.cat([beam, data['Pmu']+torch.tensor([1,0,0,0],dtype = data['Pmu'].dtype, device = data['Pmu'].device)], 1) # Use this to perturb masses if the dataset is all massless
	data['Pmu'] = torch.cat((data['Pmu'], zero_pmus), 1)
	particle_mask = data['Pmu'][...,0] != 0.
	edge_mask = particle_mask.unsqueeze(1) * particle_mask.unsqueeze(2)
	if 'scalars' in data.keys():
		data['scalars'] = torch.cat([data['scalars'][:,:2],]*(1 if c else 0) + [data['scalars']] + [0 * data['scalars'][:,[0]],] * num_particles, dim=1)
	data['Nobj'] = data['Nobj'] + 2
	data['particle_mask'] = particle_mask
	data['edge_mask'] = edge_mask
	return data


def irc_test(model, data, keys=['predict'], logg = False):
	"""
	Tests PELICAN for IR-safety and C-safety separately.
	First we create a clone of the data batch, apply expand_data() and then call the two tests, comparing the new outputs to the originals.
	"""
	if logg:
		logging.info('IRC safety test!')
 
	# First check IR safety (injection of new small momenta). This one is easier to enforce in a model -- 
	# -- simply make sure that the outputs of the network are small whenever the inputs are.

	alpha_range = np.insert(10.**np.arange(-3,0,step=2), 0, 0.)
	max_particles=2

	data_irc_copy = {key: val.clone() if type(val) is torch.Tensor else val for key, val in data.items()}
	data_irc_copy = expand_data(data_irc_copy, max_particles, c=True)
		
	outputs = model(data_irc_copy)
	outputs_ir=[]
	
	ir_results = []
	for alpha in alpha_range:
		temp = {key: [] for key in keys}
		for num_particles in range(1, max_particles + 1):
			data_ir = ir_data(data_irc_copy, num_particles, alpha)
			for key, val in model(data_ir).items(): temp[key].append(val.detach())
		outputs_ir.append([alpha, {key: torch.stack(val,0) for key, val in temp.items()}]) # stack outputs for different alpha along a new dimension for easy printing later

	for key in keys:
		ir_test = [(alpha, ((output_ir[key] - outputs[key].unsqueeze(0).repeat([max_particles]+[1]*len(outputs[key].shape)))/outputs[key]).abs().nan_to_num(0,0,0)) for (alpha, output_ir) in outputs_ir]
		ir_test = [(alpha, val[val!=0.].median()) for (alpha, val) in ir_test]
		if logg:
			logging.info(f'IR safety test MEDIAN RELATIVE deviations for key={key} (format is (order of magnitude of momenta: [1 particle, 2 particles, ...])):\nIf you see nan, that means exact zero (good!)')
		for alpha, output in ir_test:
			if logg:
				logging.warning('{:0.5g}, {}'.format(alpha, output.data.cpu().detach().numpy()))
			ir_results.append(output.data.cpu().detach().numpy())


	# Now Colinear safety (splitting a present four-momentum into two identical or almost identical halves).
	# Can also be viewed as creating 2 particles with identical spatial momenta but half of the original energy.
	# This symmetry cannot be explicitly enforced in a nonlinear network because it requires some sort of linearity in energy.
	# Instead, we simply test for it and hope that training a network improves C-safety.
	data_irc_copy = {key: val.clone() if type(val) is torch.Tensor else val for key, val in data.items()}
	data_irc_copy = expand_data(data_irc_copy, max_particles, c=True)

	data_c = c_data(data_irc_copy)
	outputs_c = model(data_c)

	c_results = []
	for key in keys:
		# if key=='weights':
		# 	c_test = ((outputs_c[key][:,:2] - outputs[key][:,:2])/outputs[key][:,:2]).abs().nan_to_num()
		# elif key=='inputs':
		# 	c_test = ((outputs_c[key][...,:2,:2] - outputs[key][...,:2,:2] )/outputs[key][...,:2,:2] ).abs().nan_to_num()
		# else:
		# 	c_test = ((outputs_c[key] - outputs[key])/outputs[key]).abs().nan_to_num()
		# if key=='inputs': breakpoint()
		c_test = ((outputs_c[key] - outputs[key])).abs().nan_to_num()
		c_test = c_test[c_test!=0.].mean()
		if logg:
			logging.info(f'C safety test MEAN ABSOLUTE deviations for key={key}: {c_test.data.cpu().detach().numpy()}')
		c_results.append(c_test.data.cpu().detach().numpy())

	ir_results = np.array(ir_results)
	c_results = np.array(c_results)
	return ir_results, c_results


def gpu_test(model, data, t0):
	logging.info('Starting the computational cost test!')
	device = model.device
	t1 = datetime.now()
	logging.info(f'{"Collation time:":<80} {(t1-t0).total_seconds()}s')
	if model.device=='gpu':
		mem_init = torch.cuda.memory_allocated(device)
		logging.info(f'{"Initial GPU memory allocation:":<80} {mem_init}')
		output = model(data)
		t2 = datetime.now()
		mem_fwd =  torch.cuda.memory_allocated(device)
		logging.info(f'{"Forward pass time:":<80} {(t2-t1).total_seconds()}s')
		logging.info(f'{"Memory after forward pass:":<80} {mem_fwd}')
		logging.info(f'{"Memory consumed by forward pass:":<80} {mem_fwd-mem_init}')
		output['predict'].sum().backward()
		t3 = datetime.now()
		mem_bwd =  torch.cuda.memory_allocated(device)
		logging.info(f'{"Backward pass time:":<80} {(t3-t2).total_seconds()}s')
		logging.info(f'{"Memory after backward pass:":<80} {mem_bwd}')
		logging.info(f'{"Total batch time:":<80} {(t3-t0).total_seconds()}s')
		del output
		with torch.no_grad:
			t4 = datetime.now()
			mem_init = torch.cuda.memory_allocated(device)
			logging.info(f'{"Initial GPU memory allocation:":<80} {mem_init}')
			output = model.forward(data)
			t5 = datetime.now()
			mem_fwd =  torch.cuda.memory_allocated(device)
			logging.info(f'{"Inference time:":<80} {(t5-t4).total_seconds()}s')
			logging.info(f'{"Inference memory usage (over initial):":<80} {mem_fwd-mem_init}')
	else:
		output = model(data)
		t2 = datetime.now()
		logging.info(f'{"Forward pass time:":<80} {(t2-t1).total_seconds()}s')
		output['predict'].sum().backward()
		t3 = datetime.now()
		logging.info(f'{"Backward pass time:":<80} {(t3-t2).total_seconds()}s')
		logging.info(f'{"Total batch time:":<80} {(t3-t0).total_seconds()}s')
		with torch.no_grad():
			t4 = datetime.now()
			output = model.forward(data)
			t5 = datetime.now()
			logging.info(f'{"Inference time:":<80} {(t5-t4).total_seconds()}s')




def tests(model, dataloader, args, tests=['permutation','batch','irc'], cov=False):
	if not args.test:
		logging.info("WARNING: network tests disabled!")
		return

	logging.info("Testing network for symmetries:")
	model.eval()
	
	t0 = datetime.now()
	data = next(iter(dataloader))
	it = iter(dataloader)
	
	for str in tests:
		if str=='gpu':
			gpu_test(model, data, t0)
		elif str=='permutation':
			with torch.no_grad(): permutation_test(model, data)
		elif str=='batch':
			with torch.no_grad(): batch_test(model, data)
		elif str=='irc':
			num = 10    # number of batches to test over
			with torch.no_grad(): 
				data1 = next(it)
				ir, c = irc_test(model, data1, keys=['predict', 'weights'] if cov else ['predict'], logg=True)
				ir = np.expand_dims(ir,0)
				c = np.expand_dims(c,0)
				for x in range(num):
					data1 = next(it)
					ir_results, c_results = irc_test(model, data1, keys=['predict', 'weights'] if cov else ['predict'], logg=False)
					ir=np.concatenate([ir,np.expand_dims(ir_results,0)])
					c=np.concatenate([c,np.expand_dims(c_results,0)])
			ir = np.mean(ir, 0)
			c = np.mean(c, 0)
			logging.info(f"ir_test_average_over_{num}_batches:")
			logging.info(f"ir: {ir}")
			logging.info(f"c: {c}")
		logging.info('Test complete!')
