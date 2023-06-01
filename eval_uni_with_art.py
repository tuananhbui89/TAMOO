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

from utils_cm import str2bool, mkdir_p, backup, writelog
from utils_arch import get_architecture
from utils_ensemble import AverageMeter
from utils_cm import member_accuracy_uniper, accuracy
import time 
import datetime

# Import ART
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import UniversalPerturbation

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
parser.add_argument('--delta', type=float, default=0.8, help='desired accuracy for universal perturbation')


parser.add_argument('--attack_type', type=str, default='pgd', help='Flag for adv attack')
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
parser.add_argument('--num_btest', type=int, default=2500, help='Number batches for testing')
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

def main_adapt_setting():
    """
    Generating universal perturbation with ART library adapted to our setting 
    """
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")

    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch, 
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_model(args.arch, args.mode, args.dataset)

    # Create the ART classifier 
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0), # min_pixel_value, max_pixel_value 
        # clip_values = None, # Set to None to make sure the perturbation are the same for all examples. see line https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/ecf15342321a8533556ff440ae56c5fef613ac7d/art/attacks/evasion/universal_perturbation.py#L210
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters(), lr=0.01), # optimizer, doesn't matter in eval
        input_shape=(3, 32, 32), # input shape, (3, 32, 32) for CIFAR-10 
        nb_classes=args.num_classes,
    )
    
    inputs = []
    targets = []
    assert(args.num_K > 1)
    # assert(args.batch % args.num_K == 0)

    # construct data 
    for i, (input, target) in enumerate(test_loader):
        inputs.append(input)
        targets.append(target)
    inputs = torch.cat(inputs, dim=0)
    targets = torch.cat(targets, dim=0)

    _, C, W, H = inputs.shape
    vlen = (inputs.shape[0] // args.num_K) * args.num_K
    inputs = torch.reshape(inputs[:vlen], [-1, args.num_K, C, W, H])
    targets = torch.reshape(targets[:vlen], [-1, args.num_K])

    attack = UniversalPerturbation(classifier, 
                                    attacker=args.attack_type, # Supported names: ‘carlini’, ‘carlini_inf’, ‘deepfool’, ‘fgsm’, ‘bim’, ‘pgd’, ‘margin’, ‘ead’, ‘newtonfool’, ‘jsma’, ‘vat’, ‘simba’.
                                    attacker_params={'eps': args.eval_epsilon, 'max_iter': args.eval_num_steps}, 
                                    delta=args.delta, # desired accuracy, fool rate >= 1 - delta 
                                    max_iter=10, # default setting in UAP paper, number to restart 
                                    eps=args.eval_epsilon, 
                                    norm=np.inf)
    
    start_time = time.time() 
    all_adv = [] 
    all_targets = []
    for i in range(inputs.shape[0]):
        if i >= args.num_btest: 
            continue
        x_test = inputs[i].numpy()
        y_test = targets[i].numpy()

        # Generate adversarial test examples
        x_test_adv = attack.generate(x=x_test)

        # Test whether the adversarial perturbation is universal 
        perturb = x_test_adv - x_test 
        delta = perturb[0] - perturb[1]
        print('perturb shape: {}'.format(perturb.shape))
        print('delta shape: {}'.format(delta.shape))
        delta = np.reshape(delta, (delta.shape[0], -1))
        delta_norm = np.linalg.norm(delta, axis=1)
        print('delta norm shape: {}'.format(delta_norm.shape))
        print("Max perturbation norm: {}".format(np.max(delta_norm)))
        print("Min perturbation norm: {}".format(np.min(delta_norm)))
        print("Max perturbation at norm linf: {}".format(np.max(np.abs(perturb))))
        print("Min perturbation at norm linf: {}".format(np.min(np.abs(perturb))))

        
        # Time for generating adversarial examples 
        end_time = time.time()
        print("Time for generating adversarial examples per batch: {} at {}/{}".format(end_time - start_time, i, inputs.shape[0]))

        # Collect all the adversarial examples 
        all_adv.append(x_test_adv)
        all_targets.append(y_test)
    
    # Concat on new axis 
    all_adv = np.stack(all_adv, axis=0) # [B, K, C, W, H]
    all_targets = np.stack(all_targets, axis=0) # [B, K]

    all_adv = torch.from_numpy(all_adv).float().cuda()
    all_targets = torch.from_numpy(all_targets).long().cuda()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    all_correct = AverageMeter()
    all_incorrect = AverageMeter()
    avg = AverageMeter()

    criterion = nn.CrossEntropyLoss()

    num_batches =  all_adv.shape[0] // args.batch 
    for i in range(num_batches): 
        # if i >= args.num_btest: 
        #     continue

        start = i*args.batch
        stop = np.min([(i+1)*args.batch, all_adv.shape[0]])  
        X_adv = all_adv[start:stop]
        cur_targets = all_targets[start:stop]

        _all_correct, _all_incorrect, _avg, _ = member_accuracy_uniper(model, X_adv, cur_targets)
        all_correct.update(_all_correct.item(), X_adv.size(0))
        all_incorrect.update(_all_incorrect.item(), X_adv.size(0))
        avg.update(_avg.item(), X_adv.size(0))

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

        if i % 10 == 0:
            writestr = 'Adv-Test: [{0}/{1}]\t'\
                    'Loss {loss.avg:.4f}\t'\
                    'Acc@1 {top1.avg:.3f}\t'\
                    'Acc@5 {top5.avg:.3f}'.format(
                i, len(test_loader), loss=losses, top1=top1, top5=top5)
            print(writestr)


    writelog("------------- UNIVERSAL PERTURBATION WITH ART LIB --------------------", logfile)
    writelog("Method: {}".format(args.method), logfile)
    writelog("Sub attack method: {}".format(args.attack_type), logfile)
    writelog("Epsilon: {}".format(args.eval_epsilon), logfile)
    writelog("Norm: {}".format(args.norm), logfile)
    writelog("Number of steps: {}".format(args.eval_num_steps), logfile)
    writelog("K: {}".format(args.num_K), logfile)
    writelog("Desired accuracy: {}".format(args.delta), logfile)
    writestr = 'top1={}, top5={}, losses={}, sar_atleast1={}, sar_all={}, sar_avg={}, sar_ens={} '.format(top1.avg, top5.avg, losses.avg, 1. - all_correct.avg, all_incorrect.avg, 1. - avg.avg, 1. - top1.avg/100.)
    writelog(writestr, logfile)
    writelog("Time for generating adversarial examples: {}".format(end_time - start_time), logfile)
    writelog("----------------------------------------", logfile)


if __name__ == "__main__":
    main_adapt_setting()
