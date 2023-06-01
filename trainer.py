import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import time
import torch.nn.functional as F
import torch.autograd as autograd
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)


from utils_ensemble import AverageMeter, accuracy, requires_grad_
from ensemble import Ensemble

from torch.autograd import Variable
import torch.optim as optim 
from pgd import PGD_Linf
from utils_cm import writelog

def Naive_Trainer(args, loader: DataLoader, models, optimizer: Optimizer,
				epoch: int, device: torch.device, writer=None, logfile=None):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	end = time.time()

	criterion = nn.CrossEntropyLoss().cuda()

	for i in range(args.num_models):
		models[i].train()
		requires_grad_(models[i], True)

	for i, (inputs, targets) in enumerate(loader):
		# measure data loading time
		data_time.update(time.time() - end)

		inputs, targets = inputs.to(device), targets.to(device)
		batch_size = inputs.size(0)
		inputs.requires_grad = True
		grads = []
		loss_std = 0
		for j in range(args.num_models):
			logits = models[j](inputs)
			loss = criterion(logits, targets)
			grad = autograd.grad(loss, inputs, create_graph=True)[0]
			grad = grad.flatten(start_dim=1)
			grads.append(grad)
			loss_std += loss


		loss = loss_std


		ensemble = Ensemble(models)
		logits = ensemble(inputs)

		# measure accuracy and record loss
		acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
		losses.update(loss.item(), batch_size)
		top1.update(acc1.item(), batch_size)
		top5.update(acc5.item(), batch_size)


		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			writestr = 'Epoch: [{0}][{1}/{2}]\t' \
					'Time {batch_time.avg:.3f}\t' \
					'Data {data_time.avg:.3f}\t' \
					'Loss {loss.avg:.4f}\t' \
					'Acc@1 {top1.avg:.3f}\t' \
					'Acc@5 {top5.avg:.3f}'.format(
				epoch, i, len(loader), batch_time=batch_time,
				data_time=data_time, loss=losses, top1=top1, top5=top5)
			writelog(writestr, logfile)


	writer.add_scalar('train/batch_time', batch_time.avg, epoch)
	writer.add_scalar('train/acc@1', top1.avg, epoch)
	writer.add_scalar('train/acc@5', top5.avg, epoch)
	writer.add_scalar('train/loss', losses.avg, epoch)

def ADV_Trainer(args, loader: DataLoader, models, optimizer: Optimizer,
				epoch: int, device: torch.device, writer=None, logfile=None):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	end = time.time()

	criterion = nn.CrossEntropyLoss().cuda()

	attack_params = dict()
	attack_params['loss_type'] = 'ce'
	attack_params['x_min'] = 0.0 
	attack_params['x_max'] = 1.0 
	attack_params['epsilon'] = args.epsilon 
	attack_params['step_size'] = args.step_size
	attack_params['num_steps'] = args.num_steps 
	attack_params['targeted'] = False
	attack_params['num_classes'] = args.num_classes
	attack_params['soft_label'] = False 
	attack_params['random_init'] = True

	for i in range(args.num_models):
		models[i].train()
		requires_grad_(models[i], True)

	for i, (inputs, targets) in enumerate(loader):

		ensemble = Ensemble(models)
		# measure data loading time
		data_time.update(time.time() - end)

		inputs, targets = inputs.to(device), targets.to(device)
		batch_size = inputs.size(0)
		inputs.requires_grad = True

		X_adv, _ = PGD_Linf(models, inputs, targets, device, attack_params)
		en_logits_adv = ensemble(X_adv)
		en_logits_nat = ensemble(inputs)
		loss_adv = criterion(en_logits_adv, targets)
		loss_nat = criterion(en_logits_nat, targets)
		loss = args.at_alpha * loss_nat + args.at_beta * loss_adv

		# measure accuracy and record loss
		acc1, acc5 = accuracy(en_logits_nat, targets, topk=(1, 5))
		losses.update(loss.item(), batch_size)
		top1.update(acc1.item(), batch_size)
		top5.update(acc5.item(), batch_size)


		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			writestr = 'Epoch: [{0}][{1}/{2}]\t' \
					'Time {batch_time.avg:.3f}\t' \
					'Data {data_time.avg:.3f}\t' \
					'Loss {loss.avg:.4f}\t' \
					'Acc@1 {top1.avg:.3f}\t' \
					'Acc@5 {top5.avg:.3f}'.format(
				epoch, i, len(loader), batch_time=batch_time,
				data_time=data_time, loss=losses, top1=top1, top5=top5)
			writelog(writestr, logfile)


	writer.add_scalar('train/batch_time', batch_time.avg, epoch)
	writer.add_scalar('train/acc@1', top1.avg, epoch)
	writer.add_scalar('train/acc@5', top5.avg, epoch)
	writer.add_scalar('train/loss', losses.avg, epoch)

def ADV_Trainer_General(attacker, args, loader: DataLoader, models, optimizer: Optimizer,
				epoch: int, device: torch.device, writer=None, logfile=None):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	end = time.time()

	criterion = nn.CrossEntropyLoss().cuda()

	attack_params = dict()
	attack_params['loss_type'] = 'ce'
	attack_params['x_min'] = 0.0 
	attack_params['x_max'] = 1.0 
	attack_params['epsilon'] = args.epsilon 
	attack_params['step_size'] = args.step_size
	attack_params['num_steps'] = args.num_steps 
	attack_params['targeted'] = False
	attack_params['num_classes'] = args.num_classes
	attack_params['soft_label'] = False 
	attack_params['random_init'] = True

	# params for moo 
	attack_params['moo_lr'] = args.moo_lr
	attack_params['moo_steps'] = args.moo_steps
	attack_params['moo_gamma'] = args.moo_gamma
	attack_params['moo_alpha'] = args.moo_alpha
	attack_params['moo_pow'] = args.moo_pow
	attack_params['m1'] = args.m1
	attack_params['m2'] = args.m2	
	attack_params['norm'] = args.norm

	for i in range(args.num_models):
		models[i].train()
		requires_grad_(models[i], True)

	for i, (inputs, targets) in enumerate(loader):
		
		ensemble = Ensemble(models)

		# measure data loading time
		data_time.update(time.time() - end)

		inputs, targets = inputs.to(device), targets.to(device)
		batch_size = inputs.size(0)
		# inputs.requires_grad = True
		X_adv, _ = attacker(models, inputs, targets, device, attack_params)
		en_logits_adv = ensemble(X_adv)
		en_logits_nat = ensemble(inputs)
		loss_adv = criterion(en_logits_adv, targets)
		loss_nat = criterion(en_logits_nat, targets)
		loss = args.at_alpha * loss_nat + args.at_beta * loss_adv

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i % args.print_freq == 0:
			# measure accuracy and record loss
			acc1, acc5 = accuracy(en_logits_nat, targets, topk=(1, 5))
			losses.update(loss.item(), batch_size)
			top1.update(acc1.item(), batch_size)
			top5.update(acc5.item(), batch_size)
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			writestr = 'Epoch: [{0}][{1}/{2}]\t' \
					'Time {batch_time.avg:.3f}\t' \
					'Data {data_time.avg:.3f}\t' \
					'Loss {loss.avg:.4f}\t' \
					'Acc@1 {top1.avg:.3f}\t' \
					'Acc@5 {top5.avg:.3f}'.format(
				epoch, i, len(loader), batch_time=batch_time,
				data_time=data_time, loss=losses, top1=top1, top5=top5)
			writelog(writestr, logfile)

			# writer.add_scalar('train/batch_time', batch_time.avg, epoch)
			# writer.add_scalar('train/acc@1', top1.avg, epoch)
			# writer.add_scalar('train/acc@5', top5.avg, epoch)
			# writer.add_scalar('train/loss', losses.avg, epoch)