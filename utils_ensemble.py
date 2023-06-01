from mimetypes import init
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)
from tqdm import tqdm
import PIL.Image
from torchvision.transforms import ToTensor
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from autoattack import AutoAttack
import foolbox as fb
import torchattacks

from ensemble import Ensemble
from utils_alg import Cosine
from utils_cm import accuracy
from utils_cm import member_accuracy
from utils_cm import member_accuracy_eot
from utils_cm import member_accuracy_uniper
from utils_alg import gradient_maps
from utils_data import change_factor

torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLOURS = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'y', 'm', 'c', 'g', 'b', 'r']
MARKERS = ['s', 'v', 'o', '*', '+', 'D', '>', 's', 'v', 'o', '*', '+', 'D', '>']
LINESTYLES = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.']
MARKERSIZE = 5
FIGSIZE = (16,8)

from utils_cm import writelog
from pgd import PGD_Linf
from pgd import PGD_Linf_ENS
from pgd import PGD_Linf_Uni
from pgd import PGD_Linf_EoT


from minmax_pt import MinMaxEns
from minmax_pt import MinMaxUni
from minmax_pt import MinMaxEoT

from moo_gd_v4 import MOOEns
from moo_gd_v4 import MOOTOEns
from moo_gd_v4 import MOOUni
from moo_gd_v4 import MOOTOUni
from moo_gd_v4 import MOOEoT 
from moo_gd_v4 import MOOTOEoT 


from mtl_pcgrad import PCGradEns
from mtl_cagrad import CAGradEns

import matplotlib.pyplot as plt 

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count



class ProgressMeter(object):
	def __init__(self, num_batches, meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def init_logfile(filename: str, text: str):
	f = open(filename, 'w')
	f.write(text+"\n")
	f.close()


def log(filename: str, text: str):
	f = open(filename, 'a')
	f.write(text+"\n")
	f.close()

def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
	for param in model.parameters():
		param.requires_grad_(requires_grad)


def copy_code(outdir):
	"""Copies files to the outdir to store complete script with each experiment"""
	# embed()
	code = []
	exclude = set([])
	for root, _, files in os.walk("./code", topdown=True):
		for f in files:
			if not f.endswith('.py'):
				continue
			code += [(root,f)]

	for r, f in code:
		codedir = os.path.join(outdir,r)
		if not os.path.exists(codedir):
			os.mkdir(codedir)
		shutil.copy2(os.path.join(r,f), os.path.join(codedir,f))
	print("Code copied to '{}'".format(outdir))


def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
	for param in model.parameters():
		param.requires_grad_(requires_grad)


def gen_plot(transmat):
	import itertools
	plt.figure(figsize=(6, 6))
	plt.yticks(np.arange(0, 3, step=1))
	plt.xticks(np.arange(0, 3, step=1))
	cmp = plt.get_cmap('Blues')
	plt.imshow(transmat, interpolation='nearest', cmap=cmp, vmin=0, vmax=100.0)
	plt.title("Transfer attack success rate")
	plt.colorbar()
	thresh = 50.0
	for i, j in itertools.product(range(transmat.shape[0]), range(transmat.shape[1])):
		plt.text(j, i, "{:0.2f}".format(transmat[i, j]),
				 horizontalalignment="center",
				 color="white" if transmat[i, j] > thresh else "black")

	plt.ylabel('Target model')
	plt.xlabel('Base model')
	buf = io.BytesIO()
	plt.savefig(buf, format='jpeg')
	buf.seek(0)
	return buf

def test(args, loader, models, epoch, device, writer=None, logfile=None):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	end = time.time()

	print_freq=100

	criterion = nn.CrossEntropyLoss().cuda()

	# switch to eval mode
	for i in range(len(models)):
		models[i].eval()

	ensemble = Ensemble(models)
	with torch.no_grad():
		for i, (inputs, targets) in enumerate(loader):
			# measure data loading time
			data_time.update(time.time() - end)
			inputs, targets = inputs.to(device), targets.to(device)

			# compute output
			outputs = ensemble(inputs)
			loss = criterion(outputs, targets)

			# measure accuracy and record loss
			acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
			losses.update(loss.item(), inputs.size(0))
			top1.update(acc1.item(), inputs.size(0))
			top5.update(acc5.item(), inputs.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % print_freq == 0:
				writestr = 'Nat-Test: [{0}/{1}]\t'\
					  'Time {batch_time.avg:.3f}\t'\
					  'Data {data_time.avg:.3f}\t'\
					  'Loss {loss.avg:.4f}\t'\
					  'Acc@1 {top1.avg:.3f}\t'\
					  'Acc@5 {top5.avg:.3f}'.format(
					i, len(loader), batch_time=batch_time, data_time=data_time,
					loss=losses, top1=top1, top5=top5)
				print(writestr)

		writestr = 'epoch={}, top1={}, top5={}, losses={}'.format(epoch, top1.avg, top5.avg, losses.avg)
		writelog(writestr, logfile)
		writer.add_scalar('loss/test', losses.avg, epoch)
		writer.add_scalar('accuracy/test@1', top1.avg, epoch)
		writer.add_scalar('accuracy/test@5', top5.avg, epoch)

def adv_test(args, loader, models, epoch, device, writer=None, logfile=None, attack_type='PGD_Linf'):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	end = time.time()
	all_correct = AverageMeter()
	all_incorrect = AverageMeter()
	avg = AverageMeter()

	single_accs = []
	for _ in range(len(models)): 
		single_accs.append(AverageMeter())

	criterion = nn.CrossEntropyLoss().cuda()

	print_freq=10

	if attack_type == 'PGD_Linf': 
		Attack = PGD_Linf
	elif attack_type == 'PGD_Linf_ENS':
		Attack = PGD_Linf_ENS
	elif attack_type == 'MinMaxEns':
		Attack = MinMaxEns
	elif attack_type == 'MOOEns': 
		Attack = MOOEns
	elif attack_type == 'MOOTOEns': 
		Attack = MOOTOEns

	# switch to eval mode
	for i in range(len(models)):
		models[i].eval()

	ensemble = Ensemble(models)

	attack_params = dict()
	attack_params['loss_type'] = args.loss_type
	attack_params['x_min'] = 0.0 
	attack_params['x_max'] = 1.0 
	attack_params['epsilon'] = args.eval_epsilon 
	attack_params['step_size'] = args.eval_step_size
	attack_params['num_steps'] = args.eval_num_steps 
	attack_params['targeted'] = False
	attack_params['num_classes'] = args.num_classes
	attack_params['soft_label'] = False 
	attack_params['random_init'] = True

	writelog('-------- GENERATING ENSEMBLE ATTACK --------', logfile)
	for key in attack_params.keys(): 
		writelog('{}={}'.format(key, attack_params[key]), logfile)

	# with torch.no_grad(): # REMOVED 
	for i, (inputs, targets) in enumerate(loader):
		# measure data loading time
		data_time.update(time.time() - end)
		inputs, targets = inputs.to(device), targets.to(device)

		X_adv, _ = Attack(models, inputs, targets, device, attack_params)

		# compute output
		outputs = ensemble(X_adv)
		loss = criterion(outputs, targets)

		# measure accuracy and record loss
		acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
		losses.update(loss.item(), X_adv.size(0))
		top1.update(acc1.item(), X_adv.size(0))
		top5.update(acc5.item(), X_adv.size(0))

		# measure accuracy of each model 
		_all_correct, _all_incorrect, _avg, _mem_accs = member_accuracy(models, X_adv, targets)
		all_correct.update(_all_correct.item(), X_adv.size(0))
		all_incorrect.update(_all_incorrect.item(), X_adv.size(0))
		avg.update(_avg.item(), X_adv.size(0))

		for _mem_acc, _meter in zip(_mem_accs, single_accs): 
			_meter.update(_mem_acc.item(), X_adv.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % print_freq == 0:
			writestr = 'Adv-Test: [{0}/{1}]\t'\
					'Time {batch_time.avg:.3f}\t'\
					'Data {data_time.avg:.3f}\t'\
					'Loss {loss.avg:.4f}\t'\
					'Acc@1 {top1.avg:.3f}\t'\
					'Acc@5 {top5.avg:.3f}'.format(
				i, len(loader), batch_time=batch_time, data_time=data_time,
				loss=losses, top1=top1, top5=top5)
			print(writestr)

	writestr = 'epoch={}, attack_type={}, top1={}, top5={}, losses={}, sar_atleast1={}, sar_all={}, sar_avg={}, sar_ens={} '.format(epoch, 
						attack_type, top1.avg, top5.avg, losses.avg, 1. - all_correct.avg, all_incorrect.avg, 1. - avg.avg, 1. - top1.avg/100.)
	for mi in range(len(single_accs)): 
		writestr += 'sar_model{}={}, '.format(mi, 1. - single_accs[mi].avg)

	writelog(writestr, logfile)
	writer.add_scalar('loss/adv-test', losses.avg, epoch)
	writer.add_scalar('accuracy/adv-test@1', top1.avg, epoch)
	writer.add_scalar('accuracy/adv-test@5', top5.avg, epoch)
	writer.add_scalar('accuracy/all_correct', all_correct.avg, epoch)
	writer.add_scalar('accuracy/all_incorrect', all_incorrect.avg, epoch)
	for mi in range(len(single_accs)):
		writer.add_scalar('accuracy/acc_model{}'.format(mi), single_accs[mi].avg, epoch)


def adv_debug(args, loader, models, epoch, device, writer=None, logfile=None, attack_type='PGD_Linf', genadv=False):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	end = time.time()
	all_correct = AverageMeter()
	all_incorrect = AverageMeter()
	avg = AverageMeter()


	single_accs = []
	for _ in range(len(models)): 
		single_accs.append(AverageMeter())

	criterion = nn.CrossEntropyLoss().cuda()

	print_freq=10

	if attack_type == 'PGD_Linf': 
		Attack = PGD_Linf
	elif attack_type == 'PGD_Linf_ENS':
		Attack = PGD_Linf_ENS
	elif attack_type == 'MinMaxEns':
		Attack = MinMaxEns
	elif attack_type == 'MOOEns': 
		Attack = MOOEns
	elif attack_type == 'MOOTOEns': 
		Attack = MOOTOEns
	elif attack_type == 'PCGradEns':
		Attack = PCGradEns
	elif attack_type == 'CAGradEns': 
		Attack = CAGradEns

	# switch to eval mode
	for i in range(len(models)):
		models[i].eval()

	ensemble = Ensemble(models)

	attack_params = dict()
	attack_params['loss_type'] = args.loss_type
	attack_params['x_min'] = 0.0 
	attack_params['x_max'] = 1.0 
	attack_params['epsilon'] = args.eval_epsilon 
	attack_params['step_size'] = args.eval_step_size
	attack_params['num_steps'] = args.eval_num_steps 
	attack_params['targeted'] = args.targeted
	attack_params['num_classes'] = args.num_classes
	attack_params['soft_label'] = False 
	attack_params['random_init'] = True
	attack_params['moo_lr'] = args.moo_lr
	attack_params['moo_steps'] = args.moo_steps
	attack_params['moo_alpha'] = args.moo_alpha
	attack_params['moo_pow'] = args.moo_pow
	attack_params['m1'] = args.m1
	attack_params['m2'] = args.m2	
	attack_params['norm'] = args.norm

	# After Softmax 
	initial_w_dict = {
		'setC_adv_ce_MOO': [0.1538630321621895, 0.1798473611474037, 0.15350095480680465, 0.5127886205911636],
		'setC_adv_ce_TAMOO': [0.1928846776485443, 0.2488998979330063, 0.19099363088607788, 0.3672217816114426],
		'setC_adv_cw_MOO': [0.07071274667978286, 0.18371461033821107, 0.052153798565268514, 0.6934188067913055],
		'setC_adv_cw_TAMOO': [0.1554445058107376, 0.29196652472019197, 0.13556667566299438, 0.41702225506305696],
		
		'setC_adv_kl_MOO': [0.1538630321621895, 0.1798473611474037, 0.15350095480680465, 0.5127886205911636],
		'setC_adv_kl_TAMOO': [0.1928846776485443, 0.2488998979330063, 0.19099363088607788, 0.3672217816114426],		
		
		'setE_adv_cw_MOO': [0.2197975680232048, 0.26171490401029585, 0.2409956082701683, 0.2774919033050537],
		'setE_adv_cw_TAMOO': [0.21766336858272553, 0.2638204336166382, 0.23723578155040742, 0.281280392408371],
		'setE_adv_ce_MOO': [0.23387848883867263, 0.27315764874219894, 0.2156560719013214, 0.27730777114629745],
		'setE_adv_ce_TAMOO': [0.23346798866987228, 0.2643063679337502, 0.228862227499485, 0.27336339056491854],

		'setC_adv_ce_C': [0.22, 0.23, 0.22, 0.33],
		'setC_adv_ce_D': [0.24, 0.25, 0.24, 0.27],

	}

	# Before Softmax 
	for key in initial_w_dict.keys(): 
		initial_w_dict[key] = [np.log(alpha) for alpha in initial_w_dict[key]]



	if args.initial_w == 'None':
		attack_params['initial_w'] = None
	elif args.initial_w in ['MOO', 'TAMOO', 'C', 'D']:
		key = '{}_{}_{}'.format(args.ens_set, args.loss_type, args.initial_w) 
		attack_params['initial_w'] = np.asarray(initial_w_dict[key])
	else: 
		raise ValueError
		

	writelog('-------- GENERATING ENSEMBLE ATTACK --------', logfile)
	for arg in vars(args):
		writelog('{}={}'.format(arg, getattr(args, arg)), logfile)
	# with torch.no_grad(): # REMOVED 

	logdata = dict()
	logdata['batch'] = dict()

	X_adves = []
	for i, (inputs, targets) in enumerate(loader):
		if i >= args.num_btest: 
			continue
		
		# measure data loading time
		data_time.update(time.time() - end)
		inputs, targets = inputs.to(device), targets.to(device)

		X_adv, _log = Attack(models, inputs, targets, device, attack_params)
		logdata['batch'][i] = _log

		# compute output
		outputs = ensemble(X_adv)
		loss = criterion(outputs, targets)

		# measure accuracy and record loss
		acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
		losses.update(loss.item(), X_adv.size(0))
		top1.update(acc1.item(), X_adv.size(0))
		top5.update(acc5.item(), X_adv.size(0))

		# measure accuracy of each model 
		_all_correct, _all_incorrect, _avg, _mem_accs = member_accuracy(models, X_adv, targets)
		all_correct.update(_all_correct.item(), X_adv.size(0))
		all_incorrect.update(_all_incorrect.item(), X_adv.size(0))
		avg.update(_avg.item(), X_adv.size(0))

		for _mem_acc, _meter in zip(_mem_accs, single_accs): 
			_meter.update(_mem_acc.item(), X_adv.size(0))

		# Generate gradient map 
		grad_map_adv_m, grad_map_adv_std = gradient_maps(models, X_adv, targets) 
		grad_map_nat_m, grad_map_nat_std = gradient_maps(models, inputs, targets)
	
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % print_freq == 0:
			writestr = 'Adv-Test: [{0}/{1}]\t'\
					'Time {batch_time.avg:.3f}\t'\
					'Data {data_time.avg:.3f}\t'\
					'Loss {loss.avg:.4f}\t'\
					'Acc@1 {top1.avg:.3f}\t'\
					'Acc@5 {top5.avg:.3f}'.format(
				i, len(loader), batch_time=batch_time, data_time=data_time,
				loss=losses, top1=top1, top5=top5)
			writelog(writestr, logfile)
		
		if genadv: 
			X_adves.append(X_adv)

	writestr = 'loss_type={}, attack_type={}, top1={}, top5={}, losses={}, sar_atleast1={}, sar_all={}, sar_avg={}, sar_ens={} '.format(args.loss_type, 
						attack_type, top1.avg, top5.avg, losses.avg, 1. - all_correct.avg, all_incorrect.avg, 1. - avg.avg, 1. - top1.avg/100.)
	for mi in range(len(single_accs)): 
		writestr += 'sar_task{}={}, '.format(mi, 1. - single_accs[mi].avg)
		logdata['sar_task={}'.format(mi)] = 1. - single_accs[mi].avg
	logdata['top1'] = top1.avg
	logdata['top5'] = top5.avg
	logdata['sar_atleast1'] = 1. - all_correct.avg
	logdata['sar_all'] = all_incorrect.avg
	logdata['sar_avg'] = 1. - avg.avg
	logdata['sar_ens'] = 1. - top1.avg/100.


	writelog(writestr, logfile)
	subtit = 'at={}_{}_eps={}_step_size={}_gamma={}_m1={}_m2={}'.format(attack_type, args.loss_type, args.eval_epsilon, args.eval_step_size, args.moo_gamma, args.m1, args.m2)

	np.save(args.save_path+'/logdata_{}.npy'.format(subtit), logdata)

	if genadv: 
		X_adves = torch.cat(X_adves, dim=0) 
		torch.save(X_adves, args.save_path+'/adv_images.pt')



def adv_ensemble(args, loader, models, epoch, device, writer=None, logfile=None, attack_type='PGD_Linf'):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	end = time.time()
	all_correct = AverageMeter()
	all_incorrect = AverageMeter()
	avg = AverageMeter()


	single_accs = []
	for _ in range(len(models)): 
		single_accs.append(AverageMeter())

	criterion = nn.CrossEntropyLoss().cuda()

	print_freq=10

	# switch to eval mode
	for i in range(len(models)):
		models[i].eval()

	ensemble = Ensemble(models)
	ensemble.eval()

	attack_params = dict()
	attack_params['loss_type'] = args.loss_type
	attack_params['x_min'] = 0.0 
	attack_params['x_max'] = 1.0 
	attack_params['epsilon'] = args.eval_epsilon 
	attack_params['step_size'] = args.eval_step_size
	attack_params['num_steps'] = args.eval_num_steps 
	attack_params['targeted'] = False
	attack_params['num_classes'] = args.num_classes
	attack_params['soft_label'] = False 
	attack_params['random_init'] = True

	writelog('-------- GENERATING ENSEMBLE ATTACK --------', logfile)
	for arg in vars(args):
		writelog('{}={}'.format(arg, getattr(args, arg)), logfile)
	# with torch.no_grad(): # REMOVED 

	for i, (inputs, targets) in enumerate(loader):
		if i >= args.num_btest: 
			continue
		
		# measure data loading time
		data_time.update(time.time() - end)
		inputs, targets = inputs.to(device), targets.to(device)

		if attack_type == 'AutoAttack':
			adversary = AutoAttack(ensemble, norm='Linf', eps=args.eval_epsilon, log_path=logfile, version='standard') 
			adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
			X_adv = adversary.run_standard_evaluation(inputs, targets, bs=100)
		elif attack_type == 'BB': 
			class init_attack(object):
        
				def __init__(self, attack):
					self.attack = attack
					
				def run(self, model, originals, criterion_):
					return self.attack(model, inputs, criterion=criterion_, epsilons=args.eval_epsilon)[1]

			pdg_init_attack = fb.attacks.LinfPGD(steps=20, abs_stepsize=args.eval_epsilon/2, random_start=True)
			bb_attack = fb.attacks.LinfinityBrendelBethgeAttack(init_attack(pdg_init_attack), steps=200)
			fmodel = fb.PyTorchModel(ensemble, bounds=(0, 1))
			_, _, init_success = pdg_init_attack(fmodel, inputs, targets, epsilons=args.eval_epsilon)
			_, X_adv, success = bb_attack(fmodel, inputs, 
										criterion=fb.criteria.Misclassification(targets), 
										epsilons=args.eval_epsilon)
		elif attack_type == 'CW': 
			cwl2_attack = torchattacks.CW(ensemble, c=1.0, kappa=0, steps=1000, lr=0.01)
			cwl2_attack.set_return_type(type='float') # Return adversarial images with float value (0-1).
			X_adv = cwl2_attack(inputs, targets)
			# cw_attack = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=9, 
			# 				steps=1000, stepsize=0.01, 
			# 				confidence=0.7, 
			# 				initial_const=0.001, 
			# 				abort_early=True)
			# fmodel = fb.PyTorchModel(ensemble, bounds=(0, 1))
			# _, X_adv, success = cw_attack(fmodel, inputs, 
			# 							criterion=fb.criteria.Misclassification(targets), 
			# 							epsilons=args.eval_epsilon)
		
		elif attack_type == 'EAD':
			ead_attack = fb.attacks.EADAttack(binary_search_steps=9, 
						steps=1000, initial_stepsize=0.01, confidence=1.0, 
						initial_const=0.001, regularization=0.01, 
						decision_rule='EN', abort_early=True)
			fmodel = fb.PyTorchModel(ensemble, bounds=(0, 1))
			_, X_adv, success = ead_attack(fmodel, inputs, 
										criterion=fb.criteria.Misclassification(targets), 
										epsilons=args.eval_epsilon)			

		else: 
			raise ValueError 

		# compute output
		outputs = ensemble(X_adv)
		loss = criterion(outputs, targets)

		# measure accuracy and record loss
		acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
		losses.update(loss.item(), X_adv.size(0))
		top1.update(acc1.item(), X_adv.size(0))
		top5.update(acc5.item(), X_adv.size(0))

		# measure accuracy of each model 
		_all_correct, _all_incorrect, _avg, _mem_accs = member_accuracy(models, X_adv, targets)
		all_correct.update(_all_correct.item(), X_adv.size(0))
		all_incorrect.update(_all_incorrect.item(), X_adv.size(0))
		avg.update(_avg.item(), X_adv.size(0))

		for _mem_acc, _meter in zip(_mem_accs, single_accs): 
			_meter.update(_mem_acc.item(), X_adv.size(0))
	
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % print_freq == 0:
			writestr = 'Adv-Test: [{0}/{1}]\t'\
					'Time {batch_time.avg:.3f}\t'\
					'Data {data_time.avg:.3f}\t'\
					'Loss {loss.avg:.4f}\t'\
					'Acc@1 {top1.avg:.3f}\t'\
					'Acc@5 {top5.avg:.3f}'.format(
				i, len(loader), batch_time=batch_time, data_time=data_time,
				loss=losses, top1=top1, top5=top5)
			writelog(writestr, logfile)

	writestr = 'loss_type={}, attack_type={}, top1={}, top5={}, losses={}, sar_atleast1={}, sar_all={}, sar_avg={}, sar_ens={} '.format(args.loss_type, 
						attack_type, top1.avg, top5.avg, losses.avg, 1. - all_correct.avg, all_incorrect.avg, 1. - avg.avg, 1. - top1.avg/100.)
	for mi in range(len(single_accs)): 
		writestr += 'sar_task{}={}, '.format(mi, 1. - single_accs[mi].avg)

	writelog(writestr, logfile)


def adv_uniper(args, loader, model, device, logfile=None, attack_type='PGD_Linf'):
	"""
	TODO: Generate Universal Perturbation 
	Args: 
		loader: data loader 
		model: a single pretrained model 

	"""
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	all_correct = AverageMeter()
	all_incorrect = AverageMeter()
	avg = AverageMeter()

	single_accs = []
	for _ in range(args.num_K): 
		single_accs.append(AverageMeter())

	end = time.time()


	criterion = nn.CrossEntropyLoss().cuda()

	print_freq=10

	if attack_type == 'PGD_Linf_Uni': 
		Attack = PGD_Linf_Uni
	elif attack_type == 'MinMaxUni':
		Attack = MinMaxUni
	elif attack_type == 'MOOUni':
		Attack = MOOUni
	elif attack_type == 'MOOTOUni':
		Attack = MOOTOUni

	# switch to eval mode
	model.eval()

	attack_params = dict()
	attack_params['loss_type'] = args.loss_type
	attack_params['x_min'] = 0.0 
	attack_params['x_max'] = 1.0 
	attack_params['epsilon'] = args.eval_epsilon 
	attack_params['step_size'] = args.eval_step_size
	attack_params['num_steps'] = args.eval_num_steps 
	attack_params['targeted'] = False
	attack_params['num_classes'] = args.num_classes
	attack_params['soft_label'] = False 
	attack_params['random_init'] = True
	attack_params['moo_lr'] = args.moo_lr
	attack_params['moo_steps'] = args.moo_steps
	attack_params['num_K'] = args.num_K
	attack_params['m1'] = args.m1
	attack_params['m2'] = args.m2	
	attack_params['norm'] = args.norm

	writelog('-------- GENERATING UNIVERSAL PERTURBATION --------', logfile)
	for arg in vars(args):
		writelog('{}={}'.format(arg, getattr(args, arg)), logfile)

	inputs = []
	targets = []
	logdata = dict()
	logdata['batch'] = dict()
	# construct data 
	for i, (input, target) in enumerate(loader):
		inputs.append(input)
		targets.append(target)
	inputs = torch.cat(inputs, dim=0)
	targets = torch.cat(targets, dim=0)

	_, C, W, H = inputs.shape
	vlen = (inputs.shape[0] // args.num_K) * args.num_K
	inputs = torch.reshape(inputs[:vlen], [-1, args.num_K, C, W, H])
	targets = torch.reshape(targets[:vlen], [-1, args.num_K])

	# start learning
	num_batches =  inputs.shape[0] // args.batch 
	for i in range(num_batches): 
		if i >= args.num_btest: 
			continue
		# measure data loading time
		data_time.update(time.time() - end)

		start = i*args.batch
		stop = np.min([(i+1)*args.batch, inputs.shape[0]])  
		cur_inputs = inputs[start:stop]
		cur_targets = targets[start:stop]
		cur_inputs, cur_targets = cur_inputs.to(device), cur_targets.to(device)
		X_adv, _log = Attack(model, cur_inputs, cur_targets, device, attack_params)

		logdata['batch'][i] = _log

		_all_correct, _all_incorrect, _avg, _mem_accs = member_accuracy_uniper(model, X_adv, cur_targets)
		all_correct.update(_all_correct.item(), X_adv.size(0))
		all_incorrect.update(_all_incorrect.item(), X_adv.size(0))
		avg.update(_avg.item(), X_adv.size(0))

		for _mem_acc, _meter in zip(_mem_accs, single_accs): 
			_meter.update(_mem_acc.item(), X_adv.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		# compute output
		X_adv = torch.reshape(X_adv, [-1, C, W, H])
		cur_targets = torch.reshape(cur_targets, [-1])
		outputs = model(X_adv)
		loss = criterion(outputs, cur_targets)

		# measure accuracy and record loss
		acc1, acc5 = accuracy(outputs, cur_targets, topk=(1, 5))
		losses.update(loss.item(), X_adv.size(0))
		top1.update(acc1.item(), X_adv.size(0))
		top5.update(acc5.item(), X_adv.size(0))

		if i % print_freq == 0:
			writestr = 'Adv-Test: [{0}/{1}]\t'\
					'Time {batch_time.avg:.3f}\t'\
					'Data {data_time.avg:.3f}\t'\
					'Loss {loss.avg:.4f}\t'\
					'Acc@1 {top1.avg:.3f}\t'\
					'Acc@5 {top5.avg:.3f}'.format(
				i, len(loader), batch_time=batch_time, data_time=data_time,
				loss=losses, top1=top1, top5=top5)
			writelog(writestr, logfile)


	writestr = 'loss_type={}, attack_type={}, top1={}, top5={}, losses={}, sar_atleast1={}, sar_all={}, sar_avg={}, sar_ens={} '.format(args.loss_type, 
						attack_type, top1.avg, top5.avg, losses.avg, 1. - all_correct.avg, all_incorrect.avg, 1. - avg.avg, 1. - top1.avg/100.)
	for mi in range(len(single_accs)): 
		writestr += 'sar_task{}={}, '.format(mi, 1. - single_accs[mi].avg)
		logdata['sar_task={}'.format(mi)] = 1. - single_accs[mi].avg
	logdata['top1'] = top1.avg
	logdata['top5'] = top5.avg
	logdata['sar_atleast1'] = 1. - all_correct.avg
	logdata['sar_all'] = all_incorrect.avg
	logdata['sar_avg'] = 1. - avg.avg
	logdata['sar_ens'] = 1. - top1.avg/100.

	writelog(writestr, logfile)
	subtit = 'at={}_{}_step_size={}_gamma={}_K={}_m1={}_m2={}'.format(attack_type, args.loss_type, args.eval_step_size, args.moo_gamma, args.num_K, args.m1, args.m2)

	np.save(args.save_path+'/logdata_{}.npy'.format(subtit), logdata)



def adv_eot(args, Trf, loader, model, epoch, device, writer=None, logfile=None, attack_type='PGD_Linf'):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	end = time.time()
	all_correct = AverageMeter()
	all_incorrect = AverageMeter()
	avg = AverageMeter()

	single_accs = []
	for _ in range(len(Trf)): 
		single_accs.append(AverageMeter())

	criterion = nn.CrossEntropyLoss().cuda()

	print_freq=10
	if attack_type == 'PGD_Linf_EoT':
		Attack = PGD_Linf_EoT
	elif attack_type == 'MinMaxEoT': 
		Attack = MinMaxEoT
	elif attack_type == 'MOOEoT':
		Attack = MOOEoT
	elif attack_type == 'MOOTOEoT':
		Attack = MOOTOEoT

	# switch to eval mode
	model.eval()

	attack_params = dict()
	attack_params['loss_type'] = args.loss_type
	attack_params['x_min'] = 0.0 
	attack_params['x_max'] = 1.0 
	attack_params['epsilon'] = args.eval_epsilon 
	attack_params['step_size'] = args.eval_step_size
	attack_params['num_steps'] = args.eval_num_steps 
	attack_params['targeted'] = args.targeted # CHANGE HERE False
	attack_params['num_classes'] = args.num_classes
	attack_params['soft_label'] = False 
	attack_params['random_init'] = True
	attack_params['moo_lr'] = args.moo_lr
	attack_params['moo_steps'] = args.moo_steps
	attack_params['moo_alpha'] = args.moo_alpha
	attack_params['moo_pow'] = args.moo_pow
	attack_params['m1'] = args.m1
	attack_params['m2'] = args.m2	
	attack_params['norm'] = args.norm
	attack_params['tau'] = args.tau
	attack_params['eotsto'] = args.eotsto
	
	writelog('-------- GENERATING ROBUST TRANSFORMATION ATTACK --------', logfile)
	for arg in vars(args):
		writelog('{}={}'.format(arg, getattr(args, arg)), logfile)

	logdata = dict()
	logdata['batch'] = dict()
	# with torch.no_grad(): # REMOVED 
	for i, (inputs, targets) in enumerate(loader):
		change_factor(args.eotsto) # Change GLOBAL_FACTOR 
		if i >= args.num_btest: 
			continue
		# measure data loading time
		data_time.update(time.time() - end)
		inputs, targets = inputs.to(device), targets.to(device)

		X_adv, _log = Attack(model, inputs, targets, Trf, device, attack_params)

		logdata['batch'][i] = _log

		# compute output
		outputs = model(X_adv)
		loss = criterion(outputs, targets)

		# measure accuracy and record loss
		acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
		losses.update(loss.item(), X_adv.size(0))
		top1.update(acc1.item(), X_adv.size(0))
		top5.update(acc5.item(), X_adv.size(0))

		# measure accuracy of each model 
		_all_correct, _all_incorrect, _avg, _mem_accs = member_accuracy_eot(model, X_adv, targets, Trf)
		all_correct.update(_all_correct.item(), X_adv.size(0))
		all_incorrect.update(_all_incorrect.item(), X_adv.size(0))
		avg.update(_avg.item(), X_adv.size(0))

		for _mem_acc, _meter in zip(_mem_accs, single_accs): 
			_meter.update(_mem_acc.item(), X_adv.size(0))
	
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % print_freq == 0:
			writestr = 'Adv-Test: [{0}/{1}]\t'\
					'Time {batch_time.avg:.3f}\t'\
					'Data {data_time.avg:.3f}\t'\
					'Loss {loss.avg:.4f}\t'\
					'Acc@1 {top1.avg:.3f}\t'\
					'Acc@5 {top5.avg:.3f}'.format(
				i, len(loader), batch_time=batch_time, data_time=data_time,
				loss=losses, top1=top1, top5=top5)
			writelog(writestr, logfile)

	writestr = 'loss_type={}, attack_type={}, top1={}, top5={}, losses={}, sar_atleast1={}, sar_all={}, sar_avg={}, sar_ens={} '.format(args.loss_type, 
						attack_type, top1.avg, top5.avg, losses.avg, 1. - all_correct.avg, all_incorrect.avg, 1. - avg.avg, 1. - top1.avg/100.)
	for mi in range(len(single_accs)): 
		writestr += 'sar_task{}={}, '.format(mi, 1. - single_accs[mi].avg)
		logdata['sar_task={}'.format(mi)] = 1. - single_accs[mi].avg
	logdata['top1'] = top1.avg
	logdata['top5'] = top5.avg
	logdata['sar_atleast1'] = 1. - all_correct.avg
	logdata['sar_all'] = all_incorrect.avg
	logdata['sar_avg'] = 1. - avg.avg
	logdata['sar_ens'] = 1. - top1.avg/100.

	writelog(writestr, logfile)
	subtit = 'at={}_{}_step_size={}_gamma={}_m1={}_m2={}'.format(attack_type, args.loss_type, args.eval_step_size, args.moo_gamma, args.m1, args.m2)

	np.save(args.save_path+'/logdata_{}.npy'.format(subtit), logdata)
