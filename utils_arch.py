import torch
import torch.nn as nn
import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from datasets import get_normalize_layer

from models.lenet import LeNet
from models.resnet import ResNet18, ResNet50
from models.preact_resnet import PreActResNet18, PreActResNet50
from models.vgg import Vgg16
from models.googlenet import GoogLeNet
from models.efficientnet import EfficientNetB0
from models.mobilenet import MobileNet
from models.wideresnet import WideResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ens_dict = {
    'WRNx2': ['wideresnet', 'wideresnet'],
    'WRNx3': ['wideresnet', 'wideresnet', 'wideresnet'],
    'WRNx4': ['wideresnet', 'wideresnet', 'wideresnet', 'wideresnet'],

    
    'R18x2': ['resnet18', 'resnet18'],
    'R18x3': ['resnet18', 'resnet18', 'resnet18'],
    'R18x4': ['resnet18', 'resnet18', 'resnet18', 'resnet18'],

    'Mobix3': ['MobileNet', 'MobileNet', 'MobileNet'],
    'Lenetx3': ['lenet', 'lenet', 'lenet'], 

    'resvggwide': ['resnet18', 'vgg16', 'wideresnet'],
    'resvggeff': ['resnet18', 'vgg16', 'efficientnet'],
    'resmooeff': ['resnet18', 'MobileNet', 'efficientnet'],
    'lemooeff': ['lenet', 'MobileNet', 'efficientnet'],

    'resnet18': ['resnet18'], 
    'wideresnet': ['wideresnet'],
    'MobileNet': ['MobileNet'], 
    'efficientnet': ['efficientnet'],
    'lenet': ['lenet'],
    'googlenet': ['googlenet'],
    'vgg16': ['vgg16']

}

def ensemble_preds(logits, mode='average_prob'):
    assert(mode == 'average_prob')
    output = 0 
    for logit in logits: 
        output += torch.softmax(logit, dim=-1) 
    
    output /= len(logits)
    output = torch.clamp(output, min=1e-40) # Important, to avoid NaN 
    return output

class EnsembleWrap(nn.Module): 
    def __init__(self, models, mode='average_prob'):
        super(EnsembleWrap, self).__init__()
        self.models = models 
        self.mode = mode
    
    def forward(self, x): 
        logits = []
        for model in self.models:
            logit = model(x)
            logits.append(logit)
        
        output = ensemble_preds(logits, self.mode)

        return output

    def parameters(self):
        # Assign parameter of ensemble to optimizer 
        param = list(self.models[0].parameters())
        for model in self.models[1:]:
            param.extend(list(model.parameters()))

        return param

def get_ensemble(arch: str, dataset: str): 
    models = []

    assert(arch in ens_dict)

    for m in ens_dict[arch]: 
        submodel = get_architecture(m, dataset)
        submodel = nn.DataParallel(submodel)
        submodel = submodel.cuda()
        models.append(submodel)
    
    return models 

def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    # assert(dataset == 'cifar10')
    if dataset == 'cifar10':
        num_classes = 10 
    elif dataset == 'cifar100':
        num_classes = 100 
    elif dataset == 'mnist': 
        num_classes = 10 
    elif dataset == 'fashionmnist':
        num_classes = 10 

    if arch == 'lenet': 
        model = LeNet(num_classes=num_classes)
    elif arch == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif arch == 'resnet50': 
        model = ResNet50(num_classes=num_classes)
    elif arch == 'preactresnet18':
        model = PreActResNet18(num_classes=num_classes)
    elif arch == 'preactresnet50':
        model = PreActResNet50(num_classes=num_classes)
    elif arch == 'vgg16':
        model = Vgg16(num_classes=num_classes)
    elif arch == 'googlenet': 
        model = GoogLeNet(num_classes=num_classes)
    elif arch == 'efficientnet': 
        model = EfficientNetB0(num_classes=num_classes)
    elif arch == 'wideresnet': 
        model = WideResNet(num_classes=num_classes)
    elif arch == 'MobileNet':
        model = MobileNet(num_classes=num_classes)
        
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model).to(device)

