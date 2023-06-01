import os 
import numpy as np 
import shutil
import datetime
import torch 
from utils_alg import wrap_loss_fn
import torch 
import random 

def set_seed(seed=20222023):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copyfile(src, dst):
    path = os.path.dirname(dst)
    mkdir_p(path)
    shutil.copyfile(src, dst)

def chdir_p(path='/content/drive/My Drive/Workspace/OT/myOT/'): 
    os.chdir(path)
    WP = os.path.dirname(os.path.realpath('__file__')) +'/'
    print('CHANGING WORKING PATH: ', WP)

def writelog(data=None, logfile=None, printlog=True, include_time=True):
    curtime = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d__%H:%M:%S%Z")
    if include_time: 
        data = curtime + '\t' + data
    fid = open(logfile,'a')
    fid.write('%s\n'%(data))
    fid.flush()
    fid.close()
    if printlog: 
        print(data)

def dict2str(d): 
    # assert(type(d)==dict)
    res = ''
    for k in d.keys(): 
        v = d[k]
        res = res + '{}:{},'.format(k,v)
    return res 

def list2str(l): 
    # assert(type(l)==list)
    res = ''
    for i in l: 
        res = res + ' {}'.format(i)
    return res 

def str2bool(v):
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def delete_existing(path, overwrite=True):
    """Delete directory if it exists

    Used for automatically rewrites existing log directories
    """
    if not overwrite:
        assert not os.path.exists(path), "Cannot overwrite {:s}".format(path)
    else:
        if os.path.exists(path):
            shutil.rmtree(path)

import glob, os, shutil

def backup(source_dir, dest_dir, filetype=".py"):
    if '.' in filetype:
        files = glob.iglob(os.path.join(source_dir, "*{}".format(filetype)))
    else: 
        files = glob.iglob(os.path.join(source_dir, "*.{}".format(filetype)))
    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, dest_dir)

def list_dir(folder_dir, filetype='.png'):
    if '.' in filetype:
        all_dir = sorted(glob.glob(folder_dir+"*"+filetype), key=os.path.getmtime)
    else:
        all_dir = sorted(glob.glob(folder_dir+"*."+filetype), key=os.path.getmtime)
    return all_dir	

def merge_dict(x, y): 
    merge = {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
    return merge


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


def member_accuracy(models, X, y): 
	"""
	E.g., given 2 models f1, f2. 
	Return: 
		f11: correct prediction by both f1, f2 
		f00: incorrect prediction by both f1, f2 
		f1: correctly predicted by f1 - normal acc
		f2: correctly predicted by f2 - normal acc 
	E.g., given 3 models f1, f2, f3 
	Return: 
		f111: correct prediction by all models
		f000: incorrect prediction by all models
		f1, f2, f3: normal acc of model f1, f2, f3
	"""

	all_res = []
	for _model in models: 
		_output = _model(X)
		_pred = torch.argmax(_output, dim=1)
		_res = _pred.eq(y) 
		_res = torch.unsqueeze(_res, dim=1).float() 
		all_res.append(_res) 
	
	all_res = torch.cat(all_res, dim=1)
	
	all_correct = torch.mean(torch.where(torch.prod(all_res, dim=1)==1,1.,0.))
	all_incorrect = torch.mean(torch.where(torch.sum(all_res, dim=1)==0,1.,0.))
	single_results = []
	for i in range(len(models)): 
		single_correct = torch.mean(all_res[:,i])
		single_results.append(single_correct)
	avg = sum(single_results)/len(single_results)
	return all_correct, all_incorrect, avg, single_results


def member_accuracy_uniper(model, X, y): 
	"""
		X: [B, K, C, W, H]
		y: [B, K]
	"""

	all_res = []
	K = X.shape[1]

	for im in range(K):
		_output = model(X[:,im]) # [B, 10]
		_pred = torch.argmax(_output, dim=1) # [B]
		_res = _pred.eq(y[:,im]) # [B] 
		_res = torch.unsqueeze(_res, dim=1).float() # [B, 1] 
		all_res.append(_res) 
	
	all_res = torch.cat(all_res, dim=1) # [B, K]
	
	all_correct = torch.mean(torch.where(torch.prod(all_res, dim=1)==1,1.,0.))
	all_incorrect = torch.mean(torch.where(torch.sum(all_res, dim=1)==0,1.,0.))
	single_results = []

	for i in range(K): 
		single_correct = torch.mean(all_res[:,i])
		single_results.append(single_correct)
		
	avg = sum(single_results)/len(single_results)
	return all_correct, all_incorrect, avg, single_results

def member_accuracy_eot(model, X, y, Trf): 
	"""
	"""

	all_res = []
	for T in Trf: 
		_output = model(T(X))
		_pred = torch.argmax(_output, dim=1)
		_res = _pred.eq(y) 
		_res = torch.unsqueeze(_res, dim=1).float() 
		all_res.append(_res) 
	
	all_res = torch.cat(all_res, dim=1)
	
	all_correct = torch.mean(torch.where(torch.prod(all_res, dim=1)==1,1.,0.))
	all_incorrect = torch.mean(torch.where(torch.sum(all_res, dim=1)==0,1.,0.))
	single_results = []
	for i in range(len(Trf)): 
		single_correct = torch.mean(all_res[:,i])
		single_results.append(single_correct)
	avg = sum(single_results)/len(single_results)
	return all_correct, all_incorrect, avg, single_results

def member_loss(models, X, X_adv, y, loss_type): 
	"""
	E.g., given 3 models f1, f2, f3 
	Return: 
		l1, l2, l3: loss of each model 
	"""

	single_losses = []
	
	for _model in models: 
		nat_logits = _model(X)
		adv_logits = _model(X_adv)
		lossi = wrap_loss_fn(y, adv_logits, nat_logits, reduction='mean', loss_type=loss_type)
		single_losses.append(lossi)

	return single_losses

def test_member_accuracy(): 
	output1 = np.asarray([[0.9, 0.1, 0.0, 0.0],[0.9, 0.1, 0.0, 0.0]])
	output2 = np.asarray([[0.9, 0.1, 0.0, 0.0],[0.1, 0.9, 0.0, 0.0]])
	output3 = np.asarray([[0.1, 0.1, 0.8, 0.0],[0.1, 0.1, 0.8, 0.0]])
	label = np.asarray([0, 1])

	output1 = torch.tensor(output1)
	output2 = torch.tensor(output2)
	output3 = torch.tensor(output3)
	label = torch.tensor(label)
	all_res = []
	for output in [output1, output2]:
		pred = torch.argmax(output, dim=1)
		res = pred.eq(label) 
		res = torch.unsqueeze(res, dim=1).float()
		all_res.append(res)
	all_res = torch.cat(all_res, dim=1)
	all_correct = torch.mean(torch.where(torch.prod(all_res, dim=1)==1,1.,0.))
	all_incorrect = torch.mean(torch.where(torch.sum(all_res, dim=1)==0,1.,0.))
	for i in range(2): 
		single_correct = torch.mean(all_res[:,i])
		print('single_correct:', single_correct)
	print('all_res:', all_res.shape, all_res)
	print('all_correct', all_correct.shape, all_correct)
	print('all_incorrect', all_incorrect.shape, all_incorrect)

if __name__ == '__main__': 
	# test_member_accuracy()
	test_uniper_reshape()