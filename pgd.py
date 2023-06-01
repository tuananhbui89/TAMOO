
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import time
import torch.nn.functional as F
import torch.autograd as autograd
import sys
import os
import numpy as np
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)


from ensemble import Ensemble
from utils_alg import Magnitude

from torch.autograd import Variable
import torch.optim as optim 

from utils_alg import random_select_target, switch_status
from utils_moo import gdquad_solv_batch_slow
from utils_moo import quad_solv_for2D_batch
from min_norm_solvers import MinNormSolver
from ensemble import weighted_ensemble 
from utils_cm import accuracy, member_accuracy 
from utils_alg import wrap_loss_fn
from utils_data import change_factor

def PGD_Linf(models, X, y, device, attack_params): 
	"""
		Reference: 
			https://github.com/yaodongyu/TRADES/blob/master/pgd_attack_cifar10.py 
			L2 attack: https://github.com/locuslab/robust_overfitting/blob/master/train_cifar.py
		Args: 
			model: pretrained model 
			X: input tensor
			y: input target 
			attack_params:
				loss_type: 'ce', 'kl' or 'mart'
				epsilon: attack boundary
				step_size: attack step size 
				num_steps: number attack step 
				order: norm order (norm l2 or linf)
				random_init: random starting point 
				x_min, x_max: range of data 
	"""
	assert(type(models) is list)

	status = "train" if models[0].training else "eval"

	log = dict()
	log['sar_all'] = []
	log['sar_atleastone'] = []
	log['sar_avg'] = []
	for im in range(len(models)): 
		log['model{}:loss'.format(im)] = []
		log['model{}:sar'.format(im)] = []


	for _model in models:
		_model.eval()

	ensemble = Ensemble(models)

	# assert(attack_params['random_init'] == True)
	# assert(attack_params['projecting'] == True)
	# assert(attack_params['order'] == np.inf)

	targeted = -1 if attack_params['targeted'] else 1 

	X_adv = Variable(X.data, requires_grad=True)

	if attack_params['random_init']:
		random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-attack_params['epsilon'], 
															attack_params['epsilon']).to(device)
		X_adv = Variable(X_adv.data + random_noise, requires_grad=True)

	if attack_params['soft_label']: 
		target = torch.argmax(ensemble(X_adv), dim=-1)
		target = target.detach()  
	else: 
		target = y 

	if attack_params['targeted']: 
		target = random_select_target(target, num_classes=attack_params["num_classes"])
	
	"""
	Note for targeted attack 
		For targeted attack: target --> to calculate loss to attack. y --> to calculate accuracy 
		For untargeted attack: target == y 
		For targeted attack: minimizing the loss w.r.t. non-true target label  
	"""


	for _ in range(attack_params['num_steps']):
		with torch.enable_grad():
			adv_logits = ensemble(X_adv)
			nat_logits = ensemble(X)
			loss = wrap_loss_fn(target, adv_logits, nat_logits, reduction='sum', loss_type=attack_params['loss_type'])
			
			sar = 100. - accuracy(adv_logits, y)[0]
			log['model{}:loss'.format(im)].append(loss.item())
			# log['model{}:sar'.format(im)].append(sar.item())
		
		all_correct, all_incorrect, acc_avg, mem_accs = member_accuracy(models, X_adv, y)
		log['sar_all'].append(100.*all_incorrect.item())
		log['sar_atleastone'].append(100. - 100.*all_correct.item())
		log['sar_avg'].append(100. - 100.*acc_avg.item())

		for im, mem_acc in enumerate(mem_accs): 
			log['model{}:sar'.format(im)].append(100. - 100.*mem_acc.item())

		if X_adv.grad is not None:
			X_adv.grad.data.zero_()
		ensemble.zero_grad()

		loss.backward()
		eta = attack_params['step_size'] * X_adv.grad.data.sign()
		X_adv = Variable(X_adv.data + targeted * eta, requires_grad=True)
		eta = torch.clamp(X_adv.data - X.data, 
							-attack_params['epsilon'], 
							attack_params['epsilon'])
		X_adv = Variable(X.data + eta, requires_grad=True)
		X_adv = Variable(torch.clamp(X_adv, 
							attack_params['x_min'], 
							attack_params['x_max']), requires_grad=True)


	for _model in models:
		switch_status(_model, status)

	X_adv = Variable(X_adv.data, requires_grad=False)
	return X_adv, log


def PGD_Linf_ENS(models, X, y, device, attack_params): 
	"""
		Reference: 
			https://github.com/yaodongyu/TRADES/blob/master/pgd_attack_cifar10.py 
			L2 attack: https://github.com/locuslab/robust_overfitting/blob/master/train_cifar.py
		Args: 
			model: pretrained model 
			X: input tensor
			y: input target 
			attack_params:
				loss_type: 'ce', 'kl' or 'mart'
				epsilon: attack boundary
				step_size: attack step size 
				num_steps: number attack step 
				order: norm order (norm l2 or linf)
				random_init: random starting point 
				x_min, x_max: range of data 
	"""
	assert(type(models) is list)

	status = "train" if models[0].training else "eval"

	log = dict()
	log['sar_all'] = []
	log['sar_atleastone'] = []
	log['sar_avg'] = []
	for im in range(len(models)): 
		log['model{}:loss'.format(im)] = []
		log['model{}:sar'.format(im)] = []

	log['norm_grad_common'] = []

	for _model in models:
		_model.eval()

	ensemble = Ensemble(models)

	# assert(attack_params['random_init'] == True)
	# assert(attack_params['projecting'] == True)
	# assert(attack_params['order'] == np.inf)

	targeted = -1 if attack_params['targeted'] else 1 

	X_adv = Variable(X.data, requires_grad=True)

	if attack_params['random_init']:
		random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-attack_params['epsilon'], 
															attack_params['epsilon']).to(device)
		X_adv = Variable(X_adv.data + random_noise, requires_grad=True)

	if attack_params['soft_label']: 
		target = torch.argmax(ensemble(X_adv), dim=-1)
		target = target.detach()  
	else: 
		target = y 

	if attack_params['targeted']: 
		target = random_select_target(target, num_classes=attack_params["num_classes"])

	"""
	Note for targeted attack 
		For targeted attack: target --> to calculate loss to attack. y --> to calculate accuracy 
		For untargeted attack: target == y 
		For targeted attack: minimizing the loss w.r.t. non-true target label  
	"""

	for _ in range(attack_params['num_steps']):
		loss = 0
		with torch.enable_grad():
			for im, _model in enumerate(models):
				adv_logits = _model(X_adv)
				nat_logits = _model(X)
				_loss = wrap_loss_fn(target, adv_logits, nat_logits, reduction='sum', loss_type=attack_params['loss_type']) 
				loss += _loss
				sar = 100. - accuracy(adv_logits, y)[0]
				log['model{}:loss'.format(im)].append(_loss.item())
				# log['model{}:sar'.format(im)].append(sar.item())

		all_correct, all_incorrect, acc_avg, mem_accs = member_accuracy(models, X_adv, y)
		log['sar_all'].append(100.*all_incorrect.item())
		log['sar_atleastone'].append(100. - 100.*all_correct.item())
		log['sar_avg'].append(100. - 100.*acc_avg.item())

		for im, mem_acc in enumerate(mem_accs): 
			log['model{}:sar'.format(im)].append(100. - 100.*mem_acc.item())

		if X_adv.grad is not None: 
			X_adv.grad.data.zero_()
		for _model in models: 
			_model.zero_grad()
		loss.backward()
		eta = attack_params['step_size'] * X_adv.grad.data.sign()
		
		X_grad_norm = X_adv.grad.data.clone()
		X_grad_norm = torch.flatten(X_grad_norm, start_dim=1)

		X_adv = Variable(X_adv.data + targeted * eta, requires_grad=True)
		eta = torch.clamp(X_adv.data - X.data, 
							-attack_params['epsilon'], 
							attack_params['epsilon'])
		X_adv = Variable(X.data + eta, requires_grad=True)
		X_adv = Variable(torch.clamp(X_adv, 
							attack_params['x_min'], 
							attack_params['x_max']), requires_grad=True)

		log['norm_grad_common'].append(torch.mean(torch.norm(X_grad_norm, p=2, dim=1), dim=0).item())

	for _model in models:
		switch_status(_model, status)

	X_adv = Variable(X_adv.data, requires_grad=False)
	return X_adv, log 


def PGD_Linf_Uni(model, X, y, device, attack_params): 
	"""
		Args: 
			model: pretrained model 
			X: input tensor
			y: input target 
			attack_params:
				loss_type: 'ce', 'kl' or 'mart'
				epsilon: attack boundary
				step_size: attack step size 
				num_steps: number attack step 
				order: norm order (norm l2 or linf)
				random_init: random starting point 
				x_min, x_max: range of data 
		PGD Attack with Multiple Objective Optimization ALTERNATIVE Solver 
		Given list of models,
		# Step 0: Init moo_weight= [1/K] * K 
		# Step 1: OUTER MAX: Fix moo_weight, find adv examples that maximize the loss 
		# Step 2: INNER MIN: fix adv examples, update moo_weight that minimize the MOO loss 
		# Back to step 1 
	"""

	status = "train" if model.training else "eval"

	log = dict()
	log['sar'] = []
	log['loss'] = []
	log['w_mean'] = []
	log['w_std'] = []
	log['max_sar'] = 0
	model.eval()

	batch_size, K, C, W, H = X.shape 

	# assert(attack_params['random_init'] == True)
	# assert(attack_params['projecting'] == True)
	# assert(attack_params['order'] == np.inf)
	# assert(attack_params['loss_type'] == 'ce')

	targeted = -1 if attack_params['targeted'] else 1 

	if attack_params['random_init']:
		delta = torch.FloatTensor(size=[batch_size, C, W, H]).uniform_(-attack_params['epsilon'], 
															attack_params['epsilon']).to(device)
	else: 
		delta = torch.zeros(size=[batch_size, C, W, H]).to(device)

	if attack_params['soft_label']:
		targets = []
		for i in range(K):
			"""
				Given a single model and K sets of samples, get predicted target of each set 
				Using one-hot encoding inside the wrap_loss_fn --> target is indince 
				return as a list 
			"""
			pred = model(X[:,i])
			target = torch.argmax(pred, dim=1)
			targets.append(target)
		targets = torch.stack(targets, dim=1)
		assert(len(targets.shape) == 2)
	else: 
		assert(len(y.shape) == 2)
		targets = y 

	if attack_params['targeted']: 
		# raise ValueError
		_targets = torch.reshape(targets, [batch_size * K,])
		targets = random_select_target(_targets, num_classes=attack_params["num_classes"])
		targets = torch.reshape(targets, [batch_size, K])

	"""
	Note for targeted attack 
		For targeted attack: target --> to calculate loss to attack. y --> to calculate accuracy 
		For untargeted attack: target == y 
		For targeted attack: minimizing the loss w.r.t. non-true target label  
	"""

	# Init moo_weight with shape [batch_size, K]
	moo_softw = 1/K * torch.ones(size=[batch_size, K]).to(X.device)

	for _ in range(attack_params['num_steps']):
		losses_avg = 0
		assert(delta.requires_grad is False)
		delta = Variable(delta.data, requires_grad=True).to(X.device)
		model.zero_grad()

		for im in range(K): 
			# Clearning previous gradient 
			with torch.enable_grad():
				adv_logits = model(X[:,im]+delta)
				nat_logits = model(X[:,im])
				loss = wrap_loss_fn(targets[:,im], adv_logits, nat_logits, reduction='none', loss_type=attack_params['loss_type'])
				losses_avg += torch.sum(moo_softw[:,im] * loss) 

		# Backprobagation 
		# clear grad in models first 
		if delta.grad is not None:
			delta.grad.data.zero_()
		model.zero_grad()
		grad = autograd.grad(losses_avg, delta)[0] 

		delta = delta + targeted * attack_params['step_size'] * torch.sign(grad)
		delta = delta.detach()

		# Project and Clip to the valid range 
		for im in range(K): 
			delta = torch.clamp(delta, -attack_params['epsilon'], attack_params['epsilon'])
			X_adv = X[:,im] + delta 
			X_adv = torch.clamp(X_adv, attack_params['x_min'], attack_params['x_max'])
			delta = X_adv - X[:,im]

		# Searching for biggest step_size that all losses were increased. Skip this step  
		temp_x = torch.reshape(X, [batch_size*K, C, W, H])
		temp_xadv = torch.reshape(X+torch.stack([delta]*K, dim=1), [batch_size*K, C, W, H])
		temp_y = torch.reshape(y, [batch_size*K])
		adv_logits = model(temp_xadv)
		nat_logits = model(temp_x)
		loss = wrap_loss_fn(temp_y, adv_logits, nat_logits, reduction='sum', loss_type=attack_params['loss_type'])
		sar = 100. - accuracy(adv_logits, temp_y)[0]
		log['loss'].append(loss.item())
		log['sar'].append(sar.item())
		log['w_mean'].append(torch.mean(torch.mean(moo_softw, dim=0), dim=0).item())
		log['w_std'].append(torch.std(torch.mean(moo_softw,dim=0), dim=0).item())
		if sar.item() > log['max_sar']: 
			log['max_sar'] = sar.item()
		delta = delta.detach()

	switch_status(model, status)

	return X + torch.stack([delta.detach()]*K, dim=1), log


def PGD_Linf_EoT(model, X, y, Trf, device, attack_params): 
	"""
		Args: 
			model: pretrained model 
			X: input tensor
			y: input target 
			Trf: list of transofrmation
			attack_params:
				loss_type: 'ce', 'kl' or 'mart'
				epsilon: attack boundary
				step_size: attack step size 
				num_steps: number attack step 
				order: norm order (norm l2 or linf)
				random_init: random starting point 
				x_min, x_max: range of data 
		PGD Attack with Multiple Objective Optimization ALTERNATIVE Solver 
		Given list of models,
		# Step 0: Init moo_weight= [1/K] * K 
		# Step 1: OUTER MAX: Fix moo_weight, find adv examples that maximize the loss 
		# Step 2: INNER MIN: fix adv examples, update moo_weight that minimize the MOO loss 
		# Back to step 1 
	"""

	status = "train" if model.training else "eval"

	log = dict()
	log['sar'] = []
	log['loss'] = []
	log['w_mean'] = []
	log['w_std'] = []
	log['max_sar'] = 0
	model.eval()

	batch_size, C, W, H = X.shape 
	K = len(Trf)

	targeted = -1 if attack_params['targeted'] else 1 

	if attack_params['random_init']:
		delta = torch.FloatTensor(size=[batch_size, C, W, H]).uniform_(-attack_params['epsilon'], 
															attack_params['epsilon']).to(device)
	else: 
		delta = torch.zeros(size=[batch_size, C, W, H]).to(device)

	if attack_params['soft_label']:
		targets = []
		for T in Trf:
			"""
				Given a single model and K sets of samples, get predicted target of each set 
				Using one-hot encoding inside the wrap_loss_fn --> target is indince 
				return as a list 
			"""
			pred = model(T(X))
			target = torch.argmax(pred, dim=1)
			targets.append(target)

	else: 
		targets = [y for _ in range(K)]

	if attack_params['targeted']: 
		# raise ValueError
		# Same targeted attack for all transformations 
		target = random_select_target(targets[0], num_classes=attack_params["num_classes"])
		targets = [target for _ in range(K)]

	"""
	Note for targeted attack 
		For targeted attack: target --> to calculate loss to attack. y --> to calculate accuracy 
		For untargeted attack: target == y 
		For targeted attack: minimizing the loss w.r.t. non-true target label  
	"""

	# Init moo_weight with shape [batch_size, K]
	moo_softw = 1/K * torch.ones(size=[batch_size, K]).to(X.device)

	for _ in range(attack_params['num_steps']):
		change_factor(attack_params['eotsto'])
		
		losses_avg = 0
		assert(delta.requires_grad is False)
		delta = Variable(delta.data, requires_grad=True).to(X.device)
		model.zero_grad()

		for im, T in enumerate(Trf): 
			# Clearning previous gradient 
			with torch.enable_grad():
				adv_logits = model(T(X+delta))
				nat_logits = model(T(X))
				loss = wrap_loss_fn(targets[im], adv_logits, nat_logits, reduction='none', loss_type=attack_params['loss_type'])
				losses_avg += torch.sum(moo_softw[:,im] * loss) 

		# Backprobagation 
		# clear grad in models first 
		if delta.grad is not None:
			delta.grad.data.zero_()
		model.zero_grad()
		grad = autograd.grad(losses_avg, delta)[0] 

		delta = delta + targeted * attack_params['step_size'] * torch.sign(grad)
		delta = delta.detach()

		# Project and Clip to the valid range 
		delta = torch.clamp(delta, -attack_params['epsilon'], attack_params['epsilon'])
		X_adv = X + delta 
		X_adv = torch.clamp(X_adv, attack_params['x_min'], attack_params['x_max'])
		delta = X_adv - X

		# Searching for biggest step_size that all losses were increased. Skip this step  
		loss = wrap_loss_fn(y, adv_logits, nat_logits, reduction='sum', loss_type=attack_params['loss_type'])
		sar = 100. - accuracy(adv_logits, y)[0]
		log['loss'].append(loss.item())
		log['sar'].append(sar.item())
		log['w_mean'].append(torch.mean(torch.mean(moo_softw, dim=0), dim=0).item())
		log['w_std'].append(torch.std(torch.mean(moo_softw,dim=0), dim=0).item())
		if sar.item() > log['max_sar']: 
			log['max_sar'] = sar.item()
		delta = delta.detach()

	switch_status(model, status)

	return X + delta, log


def RFGSM_Linf(models, X, y, device, attack_params): 
	"""
		Reference: 
			https://github.com/ftramer/ensemble-adv-training/blob/master/simple_eval.py
		Args: 
			models: pretrained models 
			X: input tensor
			y: input target 
			attack_params:
				loss_type: 'ce', 'kl' or 'mart'
				epsilon: attack boundary
				step_size: attack step size 
				num_steps: number attack step, always 1 for RFGSM 
				order: norm order (norm l2 or linf)
				random_init: random starting point 
				x_min, x_max: range of data 
	"""
	assert(type(models) is list)

	status = "train" if models[0].training else "eval"

	log = dict()
	log['sar_all'] = []
	log['sar_atleastone'] = []
	log['sar_avg'] = []
	for im in range(len(models)): 
		log['model{}:loss'.format(im)] = []
		log['model{}:sar'.format(im)] = []


	for _model in models:
		_model.eval()

	ensemble = Ensemble(models)

	# assert(attack_params['random_init'] == True)
	# assert(attack_params['projecting'] == True)
	# assert(attack_params['order'] == np.inf)

	# Ajust attack parameters based on the RFGSM paper so we can keep the code base of PGD attack 
	attack_params['num_steps'] = 1 
	attack_params['alpha'] = attack_params['epsilon']  / 2 
	attack_params['epsilon'] = attack_params['epsilon'] - attack_params['alpha']
	attack_params['step_size'] = attack_params['epsilon'] # only 1 step 
	
	targeted = -1 if attack_params['targeted'] else 1 

	X_adv = Variable(X.data, requires_grad=True)

	# Init with random noise with scale of alpha 
	# https://github.com/ftramer/ensemble-adv-training/blob/819ad7c44d7dab4712a450e35237e9e2076cf762/simple_eval.py#L52
	random_noise = torch.sign(torch.FloatTensor(*X_adv.shape).normal_(0, 1)).to(device)
	X_adv = torch.clamp(X_adv + attack_params['alpha'] * random_noise, attack_params['x_min'], attack_params['x_max'])
	X_adv = Variable(X_adv.data, requires_grad=True)


	if attack_params['soft_label']: 
		target = torch.argmax(ensemble(X_adv), dim=-1)
		target = target.detach()  
	else: 
		target = y 

	if attack_params['targeted']: 
		target = random_select_target(target, num_classes=attack_params["num_classes"])
	
	"""
	Note for targeted attack 
		For targeted attack: target --> to calculate loss to attack. y --> to calculate accuracy 
		For untargeted attack: target == y 
		For targeted attack: minimizing the loss w.r.t. non-true target label  
	"""


	for _ in range(attack_params['num_steps']):
		with torch.enable_grad():
			adv_logits = ensemble(X_adv)
			nat_logits = ensemble(X)
			loss = wrap_loss_fn(target, adv_logits, nat_logits, reduction='sum', loss_type=attack_params['loss_type'])
			
			sar = 100. - accuracy(adv_logits, y)[0]
			log['model{}:loss'.format(im)].append(loss.item())
			# log['model{}:sar'.format(im)].append(sar.item())
		
		all_correct, all_incorrect, acc_avg, mem_accs = member_accuracy(models, X_adv, y)
		log['sar_all'].append(100.*all_incorrect.item())
		log['sar_atleastone'].append(100. - 100.*all_correct.item())
		log['sar_avg'].append(100. - 100.*acc_avg.item())

		for im, mem_acc in enumerate(mem_accs): 
			log['model{}:sar'.format(im)].append(100. - 100.*mem_acc.item())

		if X_adv.grad is not None:
			X_adv.grad.data.zero_()
		ensemble.zero_grad()

		loss.backward()
		eta = attack_params['step_size'] * X_adv.grad.data.sign()
		X_adv = Variable(X_adv.data + targeted * eta, requires_grad=True)
		eta = torch.clamp(X_adv.data - X.data, 
							-attack_params['epsilon'], 
							attack_params['epsilon'])
		X_adv = Variable(X.data + eta, requires_grad=True)
		X_adv = Variable(torch.clamp(X_adv, 
							attack_params['x_min'], 
							attack_params['x_max']), requires_grad=True)


	for _model in models:
		switch_status(_model, status)

	X_adv = Variable(X_adv.data, requires_grad=False)
	return X_adv, log


def PGD_Linf_HVM(models, X, y, device, attack_params): 
	"""
		PGD with hypervolume maximization based on section 4.2 in the paper https://arxiv.org/pdf/1901.08680.pdf
		Nadir point = min loss of all models 
		Loss = sum (log (f - nadir point))

		Reference: 
			https://github.com/yaodongyu/TRADES/blob/master/pgd_attack_cifar10.py 
			L2 attack: https://github.com/locuslab/robust_overfitting/blob/master/train_cifar.py
		Args: 
			model: pretrained model 
			X: input tensor
			y: input target 
			attack_params:
				loss_type: 'ce', 'kl' or 'mart'
				epsilon: attack boundary
				step_size: attack step size 
				num_steps: number attack step 
				order: norm order (norm l2 or linf)
				random_init: random starting point 
				x_min, x_max: range of data 
	"""
	assert(type(models) is list)

	status = "train" if models[0].training else "eval"

	log = dict()
	log['sar_all'] = []
	log['sar_atleastone'] = []
	log['sar_avg'] = []
	for im in range(len(models)): 
		log['model{}:loss'.format(im)] = []
		log['model{}:sar'.format(im)] = []

	log['norm_grad_common'] = []

	for _model in models:
		_model.eval()

	ensemble = Ensemble(models)

	# assert(attack_params['random_init'] == True)
	# assert(attack_params['projecting'] == True)
	# assert(attack_params['order'] == np.inf)

	targeted = -1 if attack_params['targeted'] else 1 

	X_adv = Variable(X.data, requires_grad=True)

	if attack_params['random_init']:
		random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-attack_params['epsilon'], 
															attack_params['epsilon']).to(device)
		X_adv = Variable(X_adv.data + random_noise, requires_grad=True)

	if attack_params['soft_label']: 
		target = torch.argmax(ensemble(X_adv), dim=-1)
		target = target.detach()  
	else: 
		target = y 

	if attack_params['targeted']: 
		target = random_select_target(target, num_classes=attack_params["num_classes"])

	"""
	Note for targeted attack 
		For targeted attack: target --> to calculate loss to attack. y --> to calculate accuracy 
		For untargeted attack: target == y 
		For targeted attack: minimizing the loss w.r.t. non-true target label  
	"""

	# Init optimal weight with pre-learned weight
	opt_weight = attack_params['initial_w']
	hvm_scale = attack_params['moo_alpha'] # 0.5 # scale of hypervolume maximization 

	for _ in range(attack_params['num_steps']):
		loss = 0
		with torch.enable_grad():
			all_losses = []
			for im, _model in enumerate(models):
				adv_logits = _model(X_adv)
				nat_logits = _model(X)
				_loss = wrap_loss_fn(target, adv_logits, nat_logits, reduction='none', loss_type=attack_params['loss_type']) 

				all_losses.append(_loss)
				sar = 100. - accuracy(adv_logits, y)[0]
				log['model{}:loss'.format(im)].append(_loss.sum().item())
		
			all_losses = torch.stack(all_losses) # (num_models, batch_size) 
			all_losses = torch.transpose(all_losses, 0, 1) # (batch_size, num_models) 

			assert(all_losses.shape[1] == len(models)) 
			assert(all_losses.shape[0] == X.shape[0])

			nadir = hvm_scale * torch.min(all_losses, dim=1, keepdim=True)[0] # (batch_size, 1)
			assert(nadir.shape[1] == 1)
			assert(nadir.shape[0] == X.shape[0])

			loss = torch.sum(torch.log(all_losses - nadir), dim=1)
			loss = torch.sum(loss, dim=0) # sum over batch 


		all_correct, all_incorrect, acc_avg, mem_accs = member_accuracy(models, X_adv, y)
		log['sar_all'].append(100.*all_incorrect.item())
		log['sar_atleastone'].append(100. - 100.*all_correct.item())
		log['sar_avg'].append(100. - 100.*acc_avg.item())

		for im, mem_acc in enumerate(mem_accs): 
			log['model{}:sar'.format(im)].append(100. - 100.*mem_acc.item())

		if X_adv.grad is not None: 
			X_adv.grad.data.zero_()
		for _model in models: 
			_model.zero_grad()
		loss.backward()
		eta = attack_params['step_size'] * X_adv.grad.data.sign()
		
		X_grad_norm = X_adv.grad.data.clone()
		X_grad_norm = torch.flatten(X_grad_norm, start_dim=1)

		X_adv = Variable(X_adv.data + targeted * eta, requires_grad=True)
		eta = torch.clamp(X_adv.data - X.data, 
							-attack_params['epsilon'], 
							attack_params['epsilon'])
		X_adv = Variable(X.data + eta, requires_grad=True)
		X_adv = Variable(torch.clamp(X_adv, 
							attack_params['x_min'], 
							attack_params['x_max']), requires_grad=True)

		log['norm_grad_common'].append(torch.mean(torch.norm(X_grad_norm, p=2, dim=1), dim=0).item())

	for _model in models:
		switch_status(_model, status)

	X_adv = Variable(X_adv.data, requires_grad=False)
	return X_adv, log 