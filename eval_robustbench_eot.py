import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import random

from tensorboardX import SummaryWriter

import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from datasets import DATASETS

from utils_ensemble import adv_eot, test
from datasets import get_dataset, get_num_classes

from utils_cm import str2bool, mkdir_p, backup
from utils_arch import get_architecture
from utils_data import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, default='cifar10', choices=DATASETS)
parser.add_argument('--arch', type=str, default='resnet18')
parser.add_argument('--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=40,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--batch', default=128, type=int, metavar='N',
                    help='batchsize (default: 128)')
parser.add_argument('--num_K', default=4, type=int, metavar='N',
                    help='number of K for Universal Perturbation (default: 4)')

parser.add_argument('--epsilon', default=0.031, type=float)
parser.add_argument('--num_steps', default=10, type=int)
parser.add_argument('--step_size', default=0.007, type=float)
parser.add_argument('--eval_epsilon', default=0.031, type=float)
parser.add_argument('--eval_num_steps', default=20, type=int)
parser.add_argument('--eval_step_size', default=0.007, type=float)
parser.add_argument('--targeted', type=str2bool, default=False, help='Targeted attack or not')

parser.add_argument('--attack_type', type=str, default='PGD_Linf', help='Flag for adv attack')
parser.add_argument('--loss_type', type=str, default='ce', help='Loss type of attacks', choices=['ce','kl','cw'])
parser.add_argument('--moo_steps', type=int, default=500, help='Number of steps to solve MOO')
parser.add_argument('--moo_lr', type=float, default=0.025, help='Learning rate to solve MOO')
parser.add_argument('--moo_gamma', type=float, default=3.0, help='weight for uniformly regurlaization MOO')
parser.add_argument('--moo_alpha', type=float, default=1.0, help='weight for weight regurlaization MOO')
parser.add_argument('--moo_pow', type=float, default=1.0, help='weight for weight regurlaization MOO - power')
parser.add_argument('--at_alpha', type=float, default=1.0, help='Adv Training, param for natural loss')
parser.add_argument('--at_beta', type=float, default=1.0, help='Adv Training, param for adv loss')
parser.add_argument('--inf', type=str, default='none', help='additional information')
parser.add_argument('--log', type=str, default='none', help='additional information for log file')
parser.add_argument('--method', type=str, help='method')
parser.add_argument('--Trf', type=str, help='Transformations')
parser.add_argument('--m1', type=float, default=0.001, help='mask 1 margin')
parser.add_argument('--m2', type=float, default=0.05, help='mask 2 margin')
parser.add_argument('--num_btest', type=int, default=500, help='Number batches for testing')
parser.add_argument('--norm', type=str, default='inf', help='attack norm')
parser.add_argument('--tau', type=float, default=2.0, help='hyper param to smooth gradients')
parser.add_argument('--eotsto', type=str2bool, help='Random Augmentation in EoT setting')
parser.add_argument('--seed', default=2022, type=int, help='random seed')


parser.add_argument('--mode', type=str)

args = parser.parse_args()

assert(args.arch == 'Salman2020Do_R18')

work_path = './exp/'
parent_path = 'ds={}_arch={}_{}_Trf={}/'.format(args.dataset, args.arch, args.mode, args.Trf)
save_path = work_path + parent_path + 'method={}_eps={}_inf={}/'.format(args.method, args.eval_epsilon, args.inf) 
mkdir_p('./exp/')
mkdir_p(parent_path)
mkdir_p(save_path)
args.save_path = save_path

args.norm = np.inf if args.norm == 'inf' else int(args.norm)

args.num_classes = get_num_classes(args.dataset)

logfile = os.path.join(save_path, "eval_log={}.txt".format(args.log))

def get_Trf(Trf): 
    TrfA  = [identity, random_hflip, random_vflip, random_affine, random_perspective, random_rotate, random_resized_crop]
    TrfB = [identity, random_brightness, random_gaussianblur, random_invert]
    TrfC = TrfA + TrfB
    TrfD = [hflip, vflip, rotate, affine, center_crop, adjust_gamma, adjust_hue, adjust_saturation, adjust_brightness, adjust_contrast]
    TrfE = [identity, hflip, vflip, center_crop, adjust_gamma, adjust_brightness, rotate]
    TrfF = [identity, center_crop, rotate, hflip, vflip, adjust_brightness]

    if Trf == 'TrfA': 
        return TrfA
    elif Trf == 'TrfB': 
        return TrfB 
    elif Trf == 'TrfC': 
        return TrfC 
    elif Trf == 'TrfD':
        return TrfD
    elif Trf == 'TrfE': 
        return TrfE 
    elif Trf == 'TrfF': 
        return TrfF

from robustbench.data import load_imagenet
from robustbench.utils import load_model

def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")

    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = load_model(model_name='Salman2020Do_R18', dataset='imagenet', threat_model='Linf')
    model.eval()
    model.cuda()

    Trf = get_Trf(args.Trf)

    # test(args, test_loader, [model], epoch=None, device=device, writer=None,logfile=logfile)
    adv_eot(args, Trf, test_loader, model, epoch=None,device=device, writer=None,logfile=logfile, attack_type=args.method)
    
if __name__ == "__main__":
    main()
