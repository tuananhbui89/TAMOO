import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tensorboardX import SummaryWriter

import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from datasets import DATASETS

from utils_ensemble import adv_uniper, test, adv_test, adv_debug
from datasets import get_dataset, get_num_classes

from utils_cm import str2bool, mkdir_p, backup
from utils_arch import get_architecture

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
parser.add_argument('--m1', type=float, default=0.001, help='mask 1 margin')
parser.add_argument('--num_btest', type=int, default=500, help='Number batches for testing')
parser.add_argument('--norm', type=str, default='inf', help='attack norm')

parser.add_argument('--mode', type=str, choices=['Naive_Trainer', 'ADV_Trainer'])

args = parser.parse_args()


work_path = './exp/'
parent_path = 'ds={}_arch={}_{}_uni/'.format(args.dataset, args.arch, args.mode)
save_path = work_path + parent_path + 'method={}_eps={}_inf={}/'.format(args.method, args.eval_epsilon, args.inf) 
mkdir_p('./exp/')
mkdir_p(parent_path)
mkdir_p(save_path)
args.save_path = save_path

args.norm = np.inf if args.norm == 'inf' else int(args.norm)

args.num_classes = get_num_classes(args.dataset)

logfile = os.path.join(save_path, "eval_log={}.txt".format(args.log))

CONSTPATH = work_path+'ds={}'.format(args.dataset) + '_arch={}_num_models=1/method={}_attack=PGD_Linf_eps=0.031_inf=none/checkpoint.pth.0'

arch_dict = dict()
for arch in ['lenet', 'resnet18', 'preactresnet18', 'vgg16', 'googlenet', 'efficientnet', 'wideresnet']:
    for mode in ['Naive_Trainer', 'ADV_Trainer']: 
        arch_dict['{}:{}'.format(arch, mode)] = [arch, CONSTPATH.format(arch, mode)]


def get_model(arch, mode, dataset): 
    m = '{}:{}'.format(arch, mode)
    assert(m in arch_dict)
    submodel = get_architecture(arch_dict[m][0], dataset)
    submodel = nn.DataParallel(submodel)
    submodel.load_state_dict(torch.load(arch_dict[m][1])['state_dict'])
    submodel.eval()
    submodel = submodel.cuda()
   
    return submodel

def main():
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")

    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_model(args.arch, args.mode, args.dataset)
    adv_uniper(args, test_loader, model, device, logfile=logfile, attack_type=args.method)
    
if __name__ == "__main__":
    main()
