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

from trainer import ADV_Trainer_General
from pgd import PGD_Linf, PGD_Linf_ENS
from moo_gd_v4 import MOOEns, MOOTOEns
from minmax_pt import MinMaxEns

from utils_ensemble import test, adv_test
from datasets import get_dataset, get_num_classes
from utils_arch import get_ensemble


from utils_cm import str2bool, mkdir_p, backup, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, default='cifar10', choices=DATASETS)
parser.add_argument('--arch', type=str)
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=128, type=int, metavar='N',
                    help='batchsize (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')

parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', action='store_true',
                    help='if true, tries to resume training from existing checkpoint')
parser.add_argument('--seed', default=20222023, type=int)

parser.add_argument('--epsilon', default=0.031, type=float)
parser.add_argument('--num_steps', default=10, type=int)
parser.add_argument('--step_size', default=0.007, type=float)
parser.add_argument('--eval_epsilon', default=0.031, type=float)
parser.add_argument('--eval_num_steps', default=20, type=int)
parser.add_argument('--eval_step_size', default=0.007, type=float)

parser.add_argument('--attack_type', type=str, default='PGD_Linf', help='Flag for adv attack')
parser.add_argument('--loss_type', type=str, default='ce', help='Loss type of attacks', choices=['ce','kl'])
parser.add_argument('--moo_steps', type=int, default=500, help='Number of steps to solve MOO')
parser.add_argument('--moo_lr', type=float, default=0.1, help='Learning rate to solve MOO')
parser.add_argument('--moo_gamma', type=float, default=3.0, help='weight for uniformly regurlaization MOO')
parser.add_argument('--moo_alpha', type=float, default=1.0, help='weight for weight regurlaization MOO')
parser.add_argument('--moo_pow', type=float, default=1.0, help='weight for weight regurlaization MOO - power')
parser.add_argument('--at_alpha', type=float, default=1.0, help='Adv Training, param for natural loss')
parser.add_argument('--at_beta', type=float, default=1.0, help='Adv Training, param for adv loss')
parser.add_argument('--method', type=str, help='trainer method')
parser.add_argument('--inf', type=str, default='none', help='additional information')
parser.add_argument('--m1', type=float, default=0.001, help='mask 1 margin')
parser.add_argument('--m2', type=float, default=0.05, help='mask 2 margin')
parser.add_argument('--norm', type=str, default='inf', help='attack norm')

args = parser.parse_args()

if os.path.exists('/trainman-mount/trainman-storage-aaf306b5-9f14-4408-ab0c-b892fc356872/'):
    work_path = '/trainman-mount/trainman-storage-aaf306b5-9f14-4408-ab0c-b892fc356872/bta/AML/MOO_Ensemble/exp/'
    parent_path = 'ds={}_arch={}/'.format(args.dataset, args.arch)
    save_path = work_path + parent_path + 'method={}_attack={}_eps={}_inf={}/'.format(args.method, args.attack_type, args.epsilon, args.inf) 
    mkdir_p(work_path)
    mkdir_p(parent_path)
    mkdir_p(save_path)
    mkdir_p(save_path+'codes/')
    backup('./', save_path+'codes/')
else:
    work_path = './exp/'
    parent_path = 'ds={}_arch={}/'.format(args.dataset, args.arch)
    save_path = work_path + parent_path + 'method={}_attack={}_eps={}_inf={}/'.format(args.method, args.attack_type, args.epsilon, args.inf)  
    mkdir_p('./exp/')
    mkdir_p(parent_path)
    mkdir_p(save_path)
    mkdir_p(save_path+'codes/')
    backup('./', save_path+'codes/')

logfile = os.path.join(save_path, "train_log.txt")
model_path = os.path.join(save_path, 'checkpoint.pth')

args.num_classes = get_num_classes(args.dataset)
args.norm = np.inf if args.norm == 'inf' else int(args.norm)

def main():
    set_seed(args.seed)
    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch//2,
                             num_workers=args.workers, pin_memory=pin_memory)

    # Attacker 
    if args.method == 'PGD_Linf_ENS': 
        attacker = PGD_Linf_ENS
    elif args.method == 'PGD_Linf':
        attacker = PGD_Linf
    elif args.method == 'MOOEns':
        attacker = MOOEns
    elif args.method == 'MOOTOEns':
        attacker = MOOTOEns
    elif args.method == 'MinMaxEns': 
        attacker = MinMaxEns
    else: 
        raise ValueError 

    # Get ensemble model 
    models = get_ensemble(arch=args.arch, dataset=args.dataset)
    args.num_models = len(models)

    param = list(models[0].parameters())
    for i in range(1, args.num_models):
        param.extend(list(models[i].parameters()))

    optimizer = optim.SGD(param, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    
    writer = SummaryWriter(save_path)

    for epoch in range(args.epochs):

        # Do Adversarial Training 
        ADV_Trainer_General(attacker, args, train_loader, models, optimizer, epoch, device, writer, logfile=logfile)

        # Do evaluation 
        test(args, test_loader, models, epoch, device, writer, logfile=logfile)

        if epoch % 10 == 0 or epoch > args.epochs - 10: 
            adv_test(args, test_loader, models, epoch, device, writer, logfile=logfile)

        scheduler.step(epoch)

        if epoch % 10 == 0 or epoch > args.epochs - 10: 
            for i in range(args.num_models):
                model_path_i = model_path + "epoch={}".format(epoch) + ".%d" % (i)
                torch.save({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': models[i].state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, model_path_i)

        for _model in models: 
            _model.train()

if __name__ == "__main__":
    main()
