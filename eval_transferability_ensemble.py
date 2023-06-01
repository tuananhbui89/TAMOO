import numpy as np
import time
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

from utils_ensemble import AverageMeter
from datasets import get_dataset, get_num_classes
from utils_arch import get_architecture


from utils_cm import str2bool, mkdir_p, backup, set_seed, writelog
from utils_cm import accuracy, member_accuracy
from ensemble import Ensemble

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, default='cifar10', choices=DATASETS)
parser.add_argument('--arch', type=str)
parser.add_argument('--source_arch', type=str)
parser.add_argument('--source_attack', type=str)


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

assert(args.method == 'ADV_Trainer')
assert(args.inf == 'none')

work_path = './exp/'
parent_path = 'ds={}_arch={}/'.format(args.dataset, args.arch)
save_path = work_path + parent_path + 'method={}_attack={}_eps={}_inf={}/'.format(args.method, args.attack_type, args.epsilon, args.inf)  
mkdir_p('./exp/')
mkdir_p(parent_path)
mkdir_p(save_path)
mkdir_p(save_path+'codes/')
backup('./', save_path+'codes/')

logfile = os.path.join(save_path, "eval_log.txt")
model_path = os.path.join(save_path, 'checkpoint.pth')

args.num_classes = get_num_classes(args.dataset)
args.norm = np.inf if args.norm == 'inf' else int(args.norm)


CONSTPATH = work_path+'ds={}'.format(args.dataset) + '_arch={}_num_models=1/method={}_attack=PGD_Linf_eps=0.031_inf=none/checkpoint.pth.0'

arch_dict = dict()
for arch in ['lenet', 'resnet18', 'preactresnet18', 'vgg16', 'googlenet', 'efficientnet', 'wideresnet', 'MobileNet']:
    for mode in ['Naive_Trainer', 'ADV_Trainer']: 
        arch_dict['{}:{}'.format(arch, mode)] = [arch, CONSTPATH.format(arch, mode)]

ens_dict = {
    'setA_naive': ['lenet:Naive_Trainer', 'resnet18:Naive_Trainer', 'vgg16:Naive_Trainer'],
    'setA_adv': ['lenet:ADV_Trainer', 'resnet18:ADV_Trainer', 'vgg16:ADV_Trainer'],
    'setB_adv': ['resnet18:ADV_Trainer', 'vgg16:ADV_Trainer', 'googlenet:ADV_Trainer'],
    'setC_adv': ['resnet18:ADV_Trainer', 'vgg16:ADV_Trainer', 'googlenet:ADV_Trainer', 'efficientnet:ADV_Trainer'],
    'setC_naive': ['resnet18:Naive_Trainer', 'vgg16:Naive_Trainer', 'googlenet:Naive_Trainer', 'efficientnet:Naive_Trainer'],
    'setD_adv': ['resnet18:ADV_Trainer', 'vgg16:ADV_Trainer', 'googlenet:ADV_Trainer', 'efficientnet:ADV_Trainer', 'wideresnet:ADV_Trainer'],
    'setE_adv': ['resnet18:ADV_Trainer', 4],
    'setF_adv': ['resnet18:ADV_Trainer', 'MobileNet:ADV_Trainer', 'efficientnet:ADV_Trainer'],
    'resnet18': ['resnet18:ADV_Trainer'],
    'MobileNet': ['MobileNet:ADV_Trainer'],
    'efficientnet': ['efficientnet:ADV_Trainer'],
    'googlenet': ['googlenet:ADV_Trainer'],
    'wideresnet': ['wideresnet:ADV_Trainer'],
    'vgg16': ['vgg16:ADV_Trainer'],
    'resmooeff': ['resnet18:ADV_Trainer', 'MobileNet:ADV_Trainer', 'efficientnet:ADV_Trainer'],
    'resx4': ['resnet18:ADV_Trainer', 4],
    'resvggwide': ['resnet18:ADV_Trainer', 'vgg16:ADV_Trainer', 'wideresnet:ADV_Trainer'],
    'reseffvgg': ['resnet18:ADV_Trainer', 'efficientnet:ADV_Trainer', 'vgg16:ADV_Trainer' ],
    'mooeffvgg': ['MobileNet:ADV_Trainer', 'efficientnet:ADV_Trainer', 'vgg16:ADV_Trainer'],
    'resmoowide': ['resnet18:ADV_Trainer', 'MobileNet:ADV_Trainer', 'wideresnet:ADV_Trainer'],
    'effvggwide': ['efficientnet:ADV_Trainer', 'vgg16:ADV_Trainer', 'wideresnet:ADV_Trainer'],
    'moovggwide': ['MobileNet:ADV_Trainer', 'vgg16:ADV_Trainer', 'wideresnet:ADV_Trainer'],
    'resmooeffvgg': ['resnet18:ADV_Trainer', 'MobileNet:ADV_Trainer', 'efficientnet:ADV_Trainer', 'vgg16:ADV_Trainer'],
    'resmooeffvggwide': ['resnet18:ADV_Trainer', 'MobileNet:ADV_Trainer', 'efficientnet:ADV_Trainer', 'vgg16:ADV_Trainer', 'wideresnet:ADV_Trainer'],
}

def get_ensemble(ens_set, dataset='cifar10'): 
    models = []
    assert(ens_set in ens_dict)
    if ens_set in ['setE_adv', 'resx4']:
        return get_ensemble_same()
        
    for m in ens_dict[ens_set]: 
        submodel = get_architecture(arch_dict[m][0], dataset)
        submodel = nn.DataParallel(submodel)
        submodel.load_state_dict(torch.load(arch_dict[m][1])['state_dict'])
        submodel.eval()
        submodel = submodel.cuda()
        models.append(submodel)
        print('model {} - training: '.format(arch_dict[m][1]), submodel.training)        

    return models

def get_ensemble_same():
    models = []
    num_models = 4
    model_path = work_path + 'ds=cifar10_arch=resnet18_num_models=4/method=ADV_Trainer_attack=PGD_Linf_eps=0.031_inf=none/checkpoint.pth'
    for i in range(num_models):
        model_path_i = model_path + ".%d" % (i)
        submodel = get_architecture('resnet18', 'cifar10')
        submodel = nn.DataParallel(submodel)
        submodel.load_state_dict(torch.load(model_path_i)['state_dict'])
        submodel.eval()
        submodel = submodel.cuda()
        models.append(submodel)
        print('model {} - training: '.format(i), submodel.training)
    return models 

def main():
    set_seed(args.seed)
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    # Get ensemble model 
    models = get_ensemble(args.arch, dataset=args.dataset)
    args.num_models = len(models)
    ensemble = Ensemble(models)

    # Load adversarial examples 
    work_path = './exp/'
    parent_path = 'ds={}_ens_set={}/'.format(args.dataset, args.source_arch)
    save_path = work_path + parent_path + 'method={}_eps={}_inf=ce/'.format(args.source_attack, args.eval_epsilon)  
    adv_path = save_path + '/adv_images.pt'
    X_adves = torch.load(adv_path)

    # Do inferencce 
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    adv_losses = AverageMeter()
    adv_top1 = AverageMeter()
    adv_top5 = AverageMeter()
    end = time.time()
    all_correct = AverageMeter()
    all_incorrect = AverageMeter()
    avg = AverageMeter()


    single_accs = []
    for _ in range(len(models)): 
        single_accs.append(AverageMeter())

    print_freq=10

    criterion = nn.CrossEntropyLoss().cuda()


    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = inputs.to(device), targets.to(device)
            X_adv = X_adves[i*args.batch:i*args.batch+targets.shape[0]]
            X_adv = X_adv.to(device)
            assert(X_adv.shape == inputs.shape)

            # compute output
            outputs = ensemble(inputs)
            adv_outputs = ensemble(X_adv)

            loss = criterion(outputs, targets)
            adv_loss = criterion(adv_outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            adv_acc1, adv_acc5 = accuracy(adv_outputs, targets, topk=(1, 5))

            _all_correct, _all_incorrect, _avg, _mem_accs = member_accuracy(models, X_adv, targets)
            all_correct.update(_all_correct.item(), X_adv.size(0))
            all_incorrect.update(_all_incorrect.item(), X_adv.size(0))
            avg.update(_avg.item(), X_adv.size(0))

            for _mem_acc, _meter in zip(_mem_accs, single_accs): 
                _meter.update(_mem_acc.item(), X_adv.size(0))

            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            adv_losses.update(adv_loss.item(), inputs.size(0))
            adv_top1.update(adv_acc1.item(), inputs.size(0))
            adv_top5.update(adv_acc5.item(), inputs.size(0))
            
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
                    i, len(test_loader), batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1, top5=top5)
                print(writestr)
                writestr = 'Adv-Test: [{0}/{1}]\t'\
                        'Time {batch_time.avg:.3f}\t'\
                        'Data {data_time.avg:.3f}\t'\
                        'Loss {loss.avg:.4f}\t'\
                        'Acc@1 {top1.avg:.3f}\t'\
                        'Acc@5 {top5.avg:.3f}'.format(
                    i, len(test_loader), batch_time=batch_time, data_time=data_time,
                    loss=adv_losses, top1=adv_top1, top5=adv_top5)
                print(writestr)

        writestr = 'source_arch={}, source_attack={}, top1={}, top5={}, losses={}, adv_top1={}, adv_top5={}, adv_losses={}, sar_all={}, sar_avg={}, '.format(
            args.source_arch, args.source_attack, top1.avg, top5.avg, losses.avg, adv_top1.avg, adv_top5.avg, adv_losses.avg, all_incorrect.avg, 1. - avg.avg)
        for mi in range(len(single_accs)): 
            writestr += 'sar_task{}={}, '.format(mi, 1. - single_accs[mi].avg)

        writelog(writestr, logfile)

if __name__ == "__main__":
    main()
